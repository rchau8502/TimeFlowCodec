#if defined(__wasm__)
typedef __int128 i128;

/* Compiler runtime builtins for 128-bit arithmetic on WebAssembly */
i128 __multi3(i128 a, i128 b) {
    return a * b;
}

i128 __muloti4(i128 a, i128 b, int *overflow) {
    if (overflow) *overflow = 0;
    return a * b;
}
#endif
