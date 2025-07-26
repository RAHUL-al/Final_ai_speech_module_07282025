import { configureStore } from "@reduxjs/toolkit";
import { useDispatch, useSelector } from "react-redux";
import authSlice from "@/store/slices/authSlice";
import paragraphReducer from "@/store/slices/paragraphSlice";
import practiceSlice from "@/store/slices/practiceSlice";
import teacherReducer from "./slices/teacherSlice";
import chatReducer from "./slices/chatSlice";
import { TypedUseSelectorHook } from "react-redux";

export function makeStore() {
  return configureStore({
    reducer: {
      auth: authSlice,
      paragraph: paragraphReducer,
      practice: practiceSlice,
      teacher: teacherReducer,
      chat: chatReducer,
    },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        serializableCheck: {
          ignoredActions: ["persist/PERSIST", "persist/REHYDRATE"],
        },
      }),
    devTools: process.env.NODE_ENV !== "production",
  });
}

export const store = makeStore();

export type AppStore = ReturnType<typeof makeStore>;
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Hooks
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
