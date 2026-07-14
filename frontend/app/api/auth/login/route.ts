import { NextResponse } from "next/server";
import { setSessionCookie } from "../../../../lib/session";
import { findUserByEmail, verifyPassword } from "../../../../lib/users";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { email?: string; password?: string };
    const email = (body.email ?? "").trim().toLowerCase();
    const password = body.password ?? "";
    const user = findUserByEmail(email);
    if (!user || !verifyPassword(password, user.passwordHash)) {
      return NextResponse.json({ detail: "Invalid email or password." }, { status: 401 });
    }
    await setSessionCookie({ email: user.email, name: user.name });
    return NextResponse.json({
      user: { email: user.email, name: user.name, createdAt: user.createdAt }
    });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Login failed." },
      { status: 500 }
    );
  }
}
