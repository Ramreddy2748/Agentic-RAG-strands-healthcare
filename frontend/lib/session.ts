import { cookies } from "next/headers";

export const SESSION_COOKIE = "citemed_session";
const SESSION_TTL_SECONDS = 60 * 60 * 24 * 7;

export type SessionUser = {
  email: string;
  name: string;
};

function sessionSecret() {
  const secret = process.env.SESSION_SECRET?.trim();
  if (!secret || secret.length < 16) {
    throw new Error("SESSION_SECRET must be set to a long random value.");
  }
  return secret;
}

function toBase64Url(bytes: ArrayBuffer | Uint8Array) {
  const array = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let binary = "";
  array.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function fromBase64Url(value: string) {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/");
  const pad = padded.length % 4 === 0 ? "" : "=".repeat(4 - (padded.length % 4));
  const binary = atob(padded + pad);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function sign(payload: string) {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(sessionSecret()),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload));
  return toBase64Url(signature);
}

function timingSafeEqual(a: string, b: string) {
  if (a.length !== b.length) return false;
  let mismatch = 0;
  for (let i = 0; i < a.length; i += 1) mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return mismatch === 0;
}

export async function encodeSession(user: SessionUser) {
  const body = toBase64Url(
    new TextEncoder().encode(
      JSON.stringify({
        email: user.email,
        name: user.name,
        exp: Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS
      })
    )
  );
  return `${body}.${await sign(body)}`;
}

export async function decodeSession(token: string | undefined | null): Promise<SessionUser | null> {
  if (!token) return null;
  const [body, signature] = token.split(".");
  if (!body || !signature) return null;
  try {
    const expected = await sign(body);
    if (!timingSafeEqual(signature, expected)) return null;
    const payload = JSON.parse(new TextDecoder().decode(fromBase64Url(body))) as {
      email?: string;
      name?: string;
      exp?: number;
    };
    if (!payload.email || !payload.name || !payload.exp) return null;
    if (payload.exp < Math.floor(Date.now() / 1000)) return null;
    return { email: payload.email, name: payload.name };
  } catch {
    return null;
  }
}

export async function setSessionCookie(user: SessionUser) {
  cookies().set(SESSION_COOKIE, await encodeSession(user), {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: SESSION_TTL_SECONDS
  });
}

export function clearSessionCookie() {
  cookies().set(SESSION_COOKIE, "", {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 0
  });
}

export async function getSessionUser() {
  return decodeSession(cookies().get(SESSION_COOKIE)?.value);
}

export async function requireSessionUser() {
  const user = await getSessionUser();
  if (!user) {
    throw new Error("Authentication required.");
  }
  return user;
}
