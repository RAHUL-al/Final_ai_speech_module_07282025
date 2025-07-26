/**
 * Reusable Button component
 */
import React from "react";
import { Loader2 } from "lucide-react";
import clsx from "clsx";

const Button = ({
  variant = "primary",
  size = "md",
  loading = false,
  disabled = false,
  children,
  className,
  ...props
}) => {
  const baseClasses = "btn";

  const variantClasses = {
    primary: "btn-primary",
    secondary: "btn-secondary",
    ghost: "btn-ghost",
    danger: "bg-red-600 text-white hover:bg-red-700 active:bg-red-800",
  };

  const sizeClasses = {
    sm: "h-8 px-3 text-xs",
    md: "h-10 px-4 text-sm",
    lg: "h-12 px-6 text-base",
  };

  return (
    <button
      className={clsx(
        baseClasses,
        variantClasses[variant],
        sizeClasses[size],
        className
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
};

export default Button;
