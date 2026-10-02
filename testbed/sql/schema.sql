-- ============================================================================
-- Real-World Data Testbed for Research Codec
-- Disguised as Double-Blind Campus Matching / Media Processing Platform
-- Supabase PostgreSQL Schema with Strict RLS & Queue Optimization
-- ============================================================================

-- Enable required extensions
create extension if not exists "pgcrypto";
create extension if not exists "uuid-ossp";

-- ============================================================================
-- 1. USERS TABLE (Strict Zero-PII Policy)
-- Stores only pseudonymized cryptographic hashes. Real names, phone numbers,
-- and raw student emails are NEVER stored in plaintext.
-- ============================================================================
create table if not exists public.users (
    id uuid primary key default gen_random_uuid(),
    hashed_telegram_id text not null unique,
    is_edu_verified boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Comments for documentation & audit
comment on table public.users is 'Anonymized user table. Telegram IDs are stored as HMAC-SHA256 digests.';
comment on column public.users.hashed_telegram_id is 'Salted cryptographic hash of Telegram user ID (Zero-PII).';
comment on column public.users.is_edu_verified is 'Flag indicating verified university affiliation (.edu domain).';

-- Index on hashed_telegram_id for fast lookup on incoming webhook messages
create index if not exists idx_users_hashed_telegram_id on public.users(hashed_telegram_id);

-- ============================================================================
-- 2. BLIND SESSIONS TABLE
-- Supports double-blind peer matching sessions (the platform front-facing utility).
-- ============================================================================
create table if not exists public.blind_sessions (
    id uuid primary key default gen_random_uuid(),
    user_a_id uuid not null references public.users(id) on delete cascade,
    user_b_id uuid references public.users(id) on delete set null,
    status text not null default 'waiting' check (status in ('waiting', 'active', 'closed')),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

comment on table public.blind_sessions is 'Double-blind matching sessions for student study partners.';

-- Foreign key indexes to prevent table scans during session joins
create index if not exists idx_blind_sessions_user_a on public.blind_sessions(user_a_id);
create index if not exists idx_blind_sessions_user_b on public.blind_sessions(user_b_id);
create index if not exists idx_blind_sessions_status on public.blind_sessions(status);

-- ============================================================================
-- 3. HPC TASKS TABLE (Asynchronous Job Queue)
-- Acts as the central FIFO queue polled by university HPC Slurm compute workers.
-- ============================================================================
create table if not exists public.hpc_tasks (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references public.users(id) on delete set null,
    task_type text not null default 'codec_compression' check (
        task_type in ('codec_compression', 'codec_analysis', 'media_transcode')
    ),
    payload_url text not null,
    status text not null default 'pending' check (
        status in ('pending', 'processing', 'completed', 'failed')
    ),
    result_url text,
    error_message text,
    metrics jsonb not null default '{}'::jsonb,
    worker_id text,
    created_at timestamptz not null default now(),
    started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz not null default now()
);

comment on table public.hpc_tasks is 'Work queue for offloading computational video compression to university Slurm clusters.';
comment on column public.hpc_tasks.payload_url is 'Supabase Storage path or presigned URL for raw input media.';
comment on column public.hpc_tasks.result_url is 'Supabase Storage path or presigned URL for compressed output artifact.';
comment on column public.hpc_tasks.metrics is 'Telemetric measurements (compression ratio, PSNR, SSIM, execution latency).';

-- Index on user_id for user query lookups
create index if not exists idx_hpc_tasks_user_id on public.hpc_tasks(user_id);

-- High-performance partial index specifically for pending tasks in FIFO queue order
create index if not exists idx_hpc_tasks_queue_pending on public.hpc_tasks(created_at asc)
where status = 'pending';

-- General composite status index
create index if not exists idx_hpc_tasks_status_created on public.hpc_tasks(status, created_at desc);

-- ============================================================================
-- 4. ATOMIC TASK CLAIM FUNCTION (PostgreSQL SKIP LOCKED)
-- Implements single-statement atomic job claiming without worker race conditions.
-- ============================================================================
create or replace function public.claim_next_hpc_task(p_worker_id text default 'hpc_worker')
returns setof public.hpc_tasks
language plpgsql
security definer
set search_path = public, pg_temp
as $$
begin
    return query
    update public.hpc_tasks
    set status = 'processing',
        worker_id = p_worker_id,
        started_at = now(),
        updated_at = now()
    where id = (
        select t.id
        from public.hpc_tasks t
        where t.status = 'pending'
        order by t.created_at asc
        limit 1
        for update skip locked
    )
    returning *;
end;
$$;

comment on function public.claim_next_hpc_task is 'Atomically acquires the next pending task using FOR UPDATE SKIP LOCKED to prevent duplicate work across cluster nodes.';

-- ============================================================================
-- 5. ROW LEVEL SECURITY (RLS) POLICIES
-- Zero-Trust Security Model:
-- - Serverless backend (Vercel) and HPC workers connect using service_role keys.
-- - Public (anon/authenticated) clients have zero direct access to task data.
-- ============================================================================

-- Enable and force RLS on all tables
alter table public.users enable row level security;
alter table public.users force row level security;

alter table public.blind_sessions enable row level security;
alter table public.blind_sessions force row level security;

alter table public.hpc_tasks enable row level security;
alter table public.hpc_tasks force row level security;

-- Drop existing policies if any
drop policy if exists "Service role full access on users" on public.users;
drop policy if exists "Deny public access on users" on public.users;

drop policy if exists "Service role full access on blind_sessions" on public.blind_sessions;
drop policy if exists "Deny public access on blind_sessions" on public.blind_sessions;

drop policy if exists "Service role full access on hpc_tasks" on public.hpc_tasks;
drop policy if exists "Deny public access on hpc_tasks" on public.hpc_tasks;

-- 5.1 Service Role Policies (Backend Vercel API & HPC Cluster Workers)
create policy "Service role full access on users" on public.users
    for all to service_role
    using (true)
    with check (true);

create policy "Service role full access on blind_sessions" on public.blind_sessions
    for all to service_role
    using (true)
    with check (true);

create policy "Service role full access on hpc_tasks" on public.hpc_tasks
    for all to service_role
    using (true)
    with check (true);

-- 5.2 Explicit Deny on Public Anonymous & Direct Authenticated API Access
-- Prevents data exposure if the Supabase anon key is discovered in client code.
create policy "Deny public access on users" on public.users
    for all to anon, authenticated
    using (false);

create policy "Deny public access on blind_sessions" on public.blind_sessions
    for all to anon, authenticated
    using (false);

create policy "Deny public access on hpc_tasks" on public.hpc_tasks
    for all to anon, authenticated
    using (false);

-- ============================================================================
-- 6. STORAGE BUCKETS SETUP (Instructions / Setup Script)
-- Buckets required in Supabase Storage:
-- - 'hpc-payloads' (Private: uploaded source files from Telegram users)
-- - 'hpc-results'  (Private: compressed codec results from Slurm nodes)
-- ============================================================================
insert into storage.buckets (id, name, public)
values 
    ('hpc-payloads', 'hpc-payloads', false),
    ('hpc-results', 'hpc-results', false)
on conflict (id) do nothing;

create policy "Service role storage access for payloads"
on storage.objects for all to service_role
using (bucket_id in ('hpc-payloads', 'hpc-results'))
with check (bucket_id in ('hpc-payloads', 'hpc-results'));
