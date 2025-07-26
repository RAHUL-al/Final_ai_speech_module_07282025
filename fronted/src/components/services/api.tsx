const BASE_URL = process.env.NEXT_PUBLIC_API_URL;

interface Tokens {
  access: string;
  refresh: string;
}

export class ApiError extends Error {
  status?: number;
  data?: any;

  constructor(message: string, status?: number, data?: any) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

const getAuthToken = (): string | null => {
  if (typeof window !== "undefined") {
    return localStorage.getItem("access_token");
  }
  return null;
};

export function setTokens({ access, refresh }: Tokens): void {
  if (typeof window !== "undefined") {
    localStorage.setItem("access_token", access);
    localStorage.setItem("refresh_token", refresh);
  }
}

async function handleResponse(response: Response): Promise<any> {
  if (response.status === 401) {
    if (typeof window !== "undefined") {
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
      localStorage.removeItem("username");
      window.location.href = "/signin";
    }
    return Promise.reject(new ApiError("Session expired", 401));
  }

  if (!response.ok) {
    let errorData: any = {};
    try {
      errorData = await response.json();
    } catch (e) {
      // Ignore JSON parsing errors
    }
    const error = new ApiError(
      errorData.detail || errorData.message || "API Error",
      response.status,
      errorData
    );
    throw error;
  }

  return response.json();
}

export async function apiGet(
  endpoint: string,
  options: RequestInit = {}
): Promise<any> {
  const token = getAuthToken();
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(token && { Authorization: `Bearer ${token}` }),
    ...((options.headers as Record<string, string>) || {}),
  };

  const response = await fetch(`${BASE_URL}${endpoint}`, {
    method: "GET",
    headers,
    ...options,
  });
  return handleResponse(response);
}

export async function apiPost(
  endpoint: string,
  data: any | FormData = {},
  options: RequestInit = {}
): Promise<any> {
  const token = getAuthToken();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000);

  // Create headers object
  const headers = new Headers();

  // Only set Content-Type if it's not FormData
  if (!(data instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  // Merge custom headers
  if (options.headers) {
    const headersInit: [string, string][] =
      options.headers instanceof Headers
        ? Array.from(options.headers.entries())
        : Array.isArray(options.headers)
        ? options.headers
        : Object.entries(options.headers);

    headersInit.forEach(([key, value]) => {
      headers.append(key, value);
    });
  }

  // Prepare body
  const body = data instanceof FormData ? data : JSON.stringify(data);
  const { headers: _, ...restOptions } = options;
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: "POST",
      headers,
      body,
      ...restOptions,
      signal: controller.signal, // Add abort signal
    });
    if (!response.ok) {
      let errorData: any = {};
      try {
        errorData = await response.json();
      } catch (e) {
        // If JSON parsing fails, use status text
        errorData = { message: response.statusText };
      }

      // Create detailed error with server response
      throw new ApiError(
        errorData.detail || errorData.message || "API Error",
        response.status,
        errorData // Include full response data
      );
    }

    clearTimeout(timeoutId); // Clear timeout on success
    return handleResponse(response);
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === "AbortError") {
      throw new ApiError("Request timed out", 504);
    }
    // Re-throw ApiError instances as-is
    if (error instanceof ApiError) {
      throw error;
    }
    // Wrap other errors
    throw new ApiError(
      error instanceof Error ? error.message : "Unknown error",
      500
    );
  }
}
