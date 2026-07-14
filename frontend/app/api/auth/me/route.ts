import { NextResponse } from "next/server";
import { getSessionUser } from "../../../../lib/session";

export const runtime = "nodejs";

export async function GET() {
  const user = await getSessionUser();
  if (!user) {
    return NextResponse.json({ detail: "Not authenticated." }, { status: 401 });
  }
  return NextResponse.json({ user });
}
