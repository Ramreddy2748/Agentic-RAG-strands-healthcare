"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import { signup } from "../../lib/api";

export default function SignupPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await signup(name, email, password);
      router.replace("/");
      router.refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Signup failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-shell">
      <form className="auth-card" onSubmit={onSubmit}>
        <div className="auth-brand">
          <p className="auth-kicker">CiteMed</p>
          <h1>Create account</h1>
          <p>Sign up to upload documents and ask grounded accreditation questions.</p>
        </div>
        {error ? <div className="alert">{error}</div> : null}
        <label>
          Full name
          <input
            autoComplete="name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            required
          />
        </label>
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
            autoComplete="new-password"
            type="password"
            minLength={6}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
        </label>
        <button disabled={busy} type="submit">
          {busy ? "Creating…" : "Create account"}
        </button>
        <p className="auth-footer">
          Already registered? <Link href="/login">Sign in</Link>
        </p>
      </form>
    </main>
  );
}
