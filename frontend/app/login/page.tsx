"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { FormEvent, Suspense, useState } from "react";
import { login } from "../../lib/api";

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
      router.replace(searchParams.get("next") || "/");
      router.refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Login failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="auth-card" onSubmit={onSubmit}>
      <div className="auth-brand">
        <p className="auth-kicker">CiteMed</p>
        <h1>Sign in</h1>
        <p>Access grounded clinical answers from indexed accreditation sources.</p>
      </div>
      {error ? <div className="alert">{error}</div> : null}
      <label>
        Email
        <input
          autoComplete="email"
          type="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          required
        />
      </label>
      <label>
        Password
        <input
          autoComplete="current-password"
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          required
        />
      </label>
      <button disabled={busy} type="submit">
        {busy ? "Signing in…" : "Sign in"}
      </button>
      <p className="auth-footer">
        Need an account? <Link href="/signup">Create one</Link>
      </p>
    </form>
  );
}

export default function LoginPage() {
  return (
    <main className="auth-shell">
      <Suspense fallback={<div className="auth-card">Loading…</div>}>
        <LoginForm />
      </Suspense>
    </main>
  );
}
