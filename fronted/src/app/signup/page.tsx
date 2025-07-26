import GuestRoute from '@/components/Auth/GuestRoute';
import SignUp from '@/components/Auth/SignUp';

export default function SignUpPage() {
  return (
    <GuestRoute>
      <SignUp />
    </GuestRoute>
  );
}
