import axios from "axios";
import { store } from "@/store";
import { logoutUser } from "@/store/slices/authSlice";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL,
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      store.dispatch(logoutUser());
      window.location.href = "/signin";
    }
    return Promise.reject(error);
  }
);

export default api;
