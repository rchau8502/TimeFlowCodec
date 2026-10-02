# Asynchronous Research Testbed & Data Pipeline

A lightweight, privacy-first, asynchronous system architecture acting as a **real-world screen content testbed** for computational research video codecs (such as TimeFlowCodec). Fronted as a double-blind campus matching and high-fidelity media processing Telegram bot.

---

## 1. System Architecture

The testbed operates on a **Thin Client, Fat Backend** model:

```
+--------------------+        Webhook HTTPS POST        +-----------------------------+
|    Telegram Bot    | -------------------------------> |   Vercel Serverless API     |
| (Thin Client UI)   |                                  |     (FastAPI Python)        |
+--------------------+                                  +-----------------------------+
          |                                                            |
          | (User uploads file)                                        | 1. HMAC-SHA256 User ID
          |                                                            | 2. Upload to Storage
          v                                                            v 3. Enqueue hpc_tasks
+-------------------------------------------------------------------------------------+
|                                Supabase Backend                                     |
|  - Storage: 'hpc-payloads' (Raw media) & 'hpc-results' (TFC compressed artifacts)   |
|  - PostgreSQL: FIFO Task Queue with FOR UPDATE SKIP LOCKED atomic claim             |
|  - Row Level Security (RLS): Zero public access; service_role access only           |
+-------------------------------------------------------------------------------------+
          ^
          | (Outbound HTTPS only - No public IP required)
          | 1. Poll & claim task via SKIP LOCKED
          | 2. Download raw payload
          | 3. Run intensive Research Codec (TimeFlowCodec 1D OLS regression)
          | 4. Upload compressed artifact & post telemetry metrics
          |
+-------------------------------------------------------------------------------------+
|                      University HPC Slurm Compute Cluster                           |
|                      (hpc_worker.py via sbatch hpc_job.slurm)                       |
+-------------------------------------------------------------------------------------+
```

---

## 2. Directory Layout

```
testbed/
├── sql/
│   └── schema.sql          # Part 1: Supabase PostgreSQL Schema, RLS & SKIP LOCKED function
├── api/
│   ├── webhook.py          # Part 2: Vercel Serverless Webhook entry-point
│   └── requirements.txt    # Serverless Python dependencies
├── vercel.json             # Vercel configuration & routing rules
├── hpc/
│   ├── hpc_worker.py       # Part 3: Headless Slurm polling worker with TimeFlowCodec engine
│   ├── hpc_job.slurm       # Slurm batch submission script (sbatch)
│   └── requirements.txt    # Compute node Python dependencies
├── .env.example            # Environment variables template
└── README.md               # System architectural documentation
```

---

## 3. Privacy & Zero-PII Policy

To ensure strict compliance with academic IRB / ethics standards:
1. **No Raw Phone Numbers or Real Names:** Telegram profile names, usernames, and phone numbers are completely omitted from logs and database writes.
2. **Cryptographic Identity Hashing:** Telegram user IDs are converted to irreversible 256-bit hashes using a secret salt:
   $$\text{hashed\_id} = \text{HMAC-SHA256}(\text{Telegram\_ID}, \, \text{SECRET\_SALT})$$
3. **Academic Email Discard:** During university verification, incoming emails are matched against educational domain patterns (`.edu`, `.ac.uk`). Upon validation, the user's `is_edu_verified` flag is set to `true`, and the email string is immediately purged from serverless memory without ever touching the database.

---

## 4. Deployment Guide

### Step 1: Set Up Supabase Database & Storage
1. Navigate to your Supabase Project -> **SQL Editor**.
2. Run the script located in [`sql/schema.sql`](sql/schema.sql).
3. This creates:
   - `users`, `blind_sessions`, and `hpc_tasks` tables.
   - Strict Row-Level Security (RLS) policies granting access exclusively to the backend `service_role`.
   - The atomic queue acquisition function `claim_next_hpc_task()` utilizing `FOR UPDATE SKIP LOCKED`.
   - Private storage buckets: `hpc-payloads` and `hpc-results`.

### Step 2: Deploy Serverless Webhook to Vercel
1. Install the Vercel CLI:
   ```bash
   npm i -g vercel
   ```
2. Link and deploy from the `testbed/` directory:
   ```bash
   cd testbed
   vercel
   ```
3. Set your environment variables in the Vercel Dashboard:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_SECRET_TOKEN`
   - `TELEGRAM_SALT`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
4. Register the webhook with Telegram:
   ```bash
   curl -F "url=https://your-vercel-app.vercel.app/api/webhook" \
        -F "secret_token=your_webhook_secret_token_here_32chars" \
        https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook
   ```

### Step 3: Launch HPC Compute Worker on Slurm Cluster
1. Transfer the `hpc/` directory to your university cluster home directory:
   ```bash
   scp -r testbed/hpc username@cluster.university.edu:~/tfc_hpc
   ```
2. Create `~/.tfc_cluster.env` with restricted permissions (`chmod 600`):
   ```bash
   export SUPABASE_URL="https://your-project.supabase.co"
   export SUPABASE_SERVICE_ROLE_KEY="eyJhbGciOi..."
   export POLL_INTERVAL_SEC=5
   ```
3. Submit the Slurm batch job:
   ```bash
   cd ~/tfc_hpc
   sbatch hpc_job.slurm
   ```
4. Verify execution:
   ```bash
   squeue -u $USER
   tail -f logs/tfc_worker_*.out
   ```

---

## 5. Telemetry & Research Output

When a user sends a video or presentation clip to the bot:
1. The serverless API stages the media into `hpc-payloads` and enqueues a `pending` row.
2. The Slurm node claims the job, processes it using the **TimeFlowCodec** 1D temporal regression pipeline, and records metrics into `hpc_tasks.metrics`:
   * `compression_ratio`: Measured file reduction factor.
   * `execution_time_sec`: Compute cluster run latency.
   * `input_bytes` vs `output_bytes`: Data reduction figures.
   * `color_sampling`: 4:4:4 native RGB confirmation.
3. The compressed artifact is saved in `hpc-results`, serving as a continuous empirical dataset for codec evaluation on real-world screen and presentation streams.
