import { NextResponse } from "next/server";
import { requireSessionUser } from "../../../../lib/session";
import { errorMessage, proxyToRag } from "../../../../lib/rag-server";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    await requireSessionUser();
    const formData = await request.formData();
    const { response, payload } = await proxyToRag("/documents/upload", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      return NextResponse.json({ detail: errorMessage(payload, response.status) }, { status: response.status });
    }
    return NextResponse.json(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Upload failed.";
    const status = message === "Authentication required." ? 401 : 500;
    return NextResponse.json({ detail: message }, { status });
  }
}
