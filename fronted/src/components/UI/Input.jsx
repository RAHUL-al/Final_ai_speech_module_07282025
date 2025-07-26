/**
 * Reusable Input component
 */
import React from "react";
 import clsx from "clsx";

const Input = React.forwardRef(
  ({ label, error, helperText, className, ...props }, ref) => {
    return (
      <div className="space-y-1">
        {label && (
          <label className="block text-sm font-medium text-gray-700">
            {label}
            {props.required && <span className="text-red-500 ml-1">*</span>}
          </label>
        )}

        <input
          ref={ref}
          className={clsx(
            "input",
            error && "border-red-300 focus:ring-red-500",
            className
          )}
          {...props}
        />

        {error && <p className="text-sm text-red-600">{error}</p>}

        {helperText && !error && (
          <p className="text-sm text-gray-500">{helperText}</p>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";

export default Input;
