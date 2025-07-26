import React from "react";

interface ProgressProps {
  value: number;
  className?: string;
  indicatorClassName?: string;
}

const Progress: React.FC<ProgressProps> = ({
  value,
  className = "",
  indicatorClassName,
}) => {
  const progressValue = Math.min(100, Math.max(0, value));

  return (
    <div
      className={`w-full bg-gray-200 rounded-full overflow-hidden ${className}`}
    >
      <div
        className="h-2 bg-blue-500 transition-all duration-300 ease-in-out"
        style={{ width: `${progressValue}%` }}
      />
    </div>
  );
};

export default Progress;
