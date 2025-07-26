import ProtectedRoute from "@/components/Auth/ProtectedRoute";
import Teacher from "@/components/Teacher/Teacher";

export default function DashboardWrapper() {
  return (
    <ProtectedRoute>
      <Teacher />
    </ProtectedRoute>
  );
}
