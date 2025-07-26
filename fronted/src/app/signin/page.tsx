'use client';

import GuestRoute from '@/components/Auth/GuestRoute';
import SignIn from '@/components/Auth/SignIn';

export default function SignInPage() {
  return (
    <GuestRoute>
      <SignIn />
    </GuestRoute>
  );
}
