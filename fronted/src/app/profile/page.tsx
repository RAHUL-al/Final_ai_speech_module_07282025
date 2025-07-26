import ProtectedRoute from "@/components/Auth/ProtectedRoute";
import Layout from "@/components/Layout/Layout";
import ProfilePage from "@/components/Profile/ProfilePage";

export default function ProfileWrapper() {
  return (
    <ProtectedRoute>
      <Layout>
        <ProfilePage />
      </Layout>
    </ProtectedRoute>
  );
}
