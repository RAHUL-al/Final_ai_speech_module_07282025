import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { ApiError, apiPost } from "@/components/services/api";
import axios from "axios";

// const getInitialAuthState = () => {
//   const token = localStorage.getItem("access_token");
//   // const refreshToken = localStorage.getItem("refresh_token");

//   return {
//     user: null,
//     tokens: {
//       access: localStorage.getItem("access_token"),
//     },
//     username: localStorage.getItem("username"),
//     isAuthenticated: !!token,
//     isLoading: !!token,
//     error: null,
//   };
// };

interface LoginCredentials {
  username: string;
  password: string;
}

interface RegisterParams {
  email: string;
  password: string;
  confirm_password: string;
  username: string;
}

interface AuthResponse {
  user: User;
  tokens: Tokens;
}

interface AuthError {
  message: string;
  code?: number;
}

interface Tokens {
  access_token: string;
  refresh_token?: string;
}

interface User {
  id: string;
  username: string;
  email: string;
  // Add other user properties as needed
}

interface AuthError {
  message: string;
  code?: number;
}

interface AuthState {
  user: User | null;
  tokens: Tokens | null;
  username: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: AuthError | null;
}

const initialState: AuthState = {
  user: null,
  tokens: null,
  username: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
};

export const loginUser = createAsyncThunk(
  "/login",
  async (credentials: LoginCredentials, { rejectWithValue }) => {
    try {
      const response = await apiPost("/login", credentials);
      return response;
    } catch (error) {
      console.error("Auth slice: Login error:", error);

      // Handle ApiError instances
      if (error instanceof ApiError) {
        return rejectWithValue({
          message: error.data?.detail || error.message,
          code: error.status,
        });
      }

      // Handle other errors
      return rejectWithValue({
        message: error instanceof Error ? error.message : "Login failed",
        code: 500,
      });
    }
  }
);

export const registerUser = createAsyncThunk<
  AuthResponse,
  RegisterParams,
  { rejectValue: AuthError }
>("auth/register", async (data, { rejectWithValue }) => {
  try {
    const response = await apiPost("/register", data);
    return response as AuthResponse;
  } catch (error) {
    if (error instanceof ApiError) {
      // Use the detailed error message from server response
      return rejectWithValue({
        message: error.data?.detail || error.message,
        code: error.status,
      });
    }
    // Handle non-ApiError cases
    return rejectWithValue({
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

export const logoutUser = createAsyncThunk(
  "/logout",
  async (_, { rejectWithValue }) => {
    try {
      await apiPost("/logout");
      return undefined;
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || "Logout failed");
    }
  }
);

export const verifyAuth = createAsyncThunk(
  "/verifyAuth",
  async (_, { rejectWithValue }) => {
    try {
      const token = localStorage.getItem("access_token");
      console.log(token);

      return token ? true : false;
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || "Logout failed");
    }
  }
);

const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    setTokens: (state, action) => {
      state.tokens = action.payload;
      state.isAuthenticated = true;
      localStorage.setItem("access_token", action.payload.access_token);
    },
    clearAuth: (state) => {
      state.user = null;
      state.tokens = null;
      state.isAuthenticated = false;
      state.error = null;
      localStorage.removeItem("access_token");
    },
    setUser: (state, action) => {
      state.user = action.payload;
    },
  },
  extraReducers: (builder) => {
    // Login
    builder
      .addCase(loginUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload.user;
        state.tokens = action.payload.tokens;
        state.isAuthenticated = true;
        state.error = null;
        localStorage.setItem("access_token", action.payload.access_token);
        localStorage.setItem("username", action.payload.username);
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as unknown as any;
        state.isAuthenticated = false;
        localStorage.removeItem("access_token");
      });

    // Register
    builder
      .addCase(registerUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(registerUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload.user;
        state.tokens = action.payload.tokens;
        state.isAuthenticated = false;
        state.error = null;
      })
      .addCase(registerUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload;
        state.isAuthenticated = false;
        localStorage.removeItem("access_token");
      });

    // Logout
    builder
      .addCase(logoutUser.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(logoutUser.fulfilled, (state) => {
        state.isLoading = false;
        state.user = null;
        state.tokens = null;
        state.isAuthenticated = false;
        state.error = null;
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        localStorage.removeItem("username");
      })
      .addCase(logoutUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as unknown as any;
        state.user = null;
        state.tokens = null;
        state.isAuthenticated = false;
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        localStorage.removeItem("username");
      });

    // Logout
    builder
      .addCase(verifyAuth.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(verifyAuth.fulfilled, (state, action) => {
        state.isLoading = false;
        state.isAuthenticated = action.payload;
      })
      .addCase(verifyAuth.rejected, (state) => {
        state.isAuthenticated = false;
      });
  },
});

export const { clearError, setTokens, clearAuth, setUser } = authSlice.actions;

export default authSlice.reducer;
