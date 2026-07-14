import { NextResponse } from "next/server";
import { setSessionCookie } from "../../../../lib/session";
import { createUser } from "../../../../lib/users";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      email?: string;
      name?: string;
      password?: string;
    };
    const user = createUser({
      email: body.email ?? "",
      name: body.name ?? "",
      password: body.password ?? ""
    });
    await setSessionCookie({ email: user.email, name: user.name });
    return NextResponse.json({ user });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Signup failed." },
      { status: 400 }
    );
  }
}
