import React, { useState } from "react";
import { Typography } from "@material-tailwind/react";
import RoleThread from "./RoleThread";

const RoleThreads = ({ threads = [] }) => {
  // Default threads to an empty array if undefined
  const [selectedThreadName, setSelectedThreadName] = useState(
    threads[0]?.name || ""
  );

  const getSelectedThreadMessages = () => {
    const selectedThread = threads.find(
      (thread) => thread.name === selectedThreadName
    );
    return selectedThread?.messages || [];
  };
  const getSelectedThread = () => {
    const selectedThread = threads.find(
      (thread) => thread.name === selectedThreadName
    );
    return selectedThread;
  };

  return (
    <div className="flex flex-col mt-4 overflow-y-scroll">
      <div className="flex flex-row gap-6 justify-center mb-2">
        {threads.map((thread) => (
          <Typography
            variant="paragraph"
            className={`font-sm mr-2 p-1 rounded-xl cursor-pointer ${
              selectedThreadName === thread.name
                ? "bg-blue-gray-50"
                : "hover:bg-blue-gray-50"
            }`}
            key={thread.name}
            onClick={() => setSelectedThreadName(thread.name)}
          >
            #{thread.name}
          </Typography>
        ))}
      </div>
      <RoleThread data={getSelectedThread()} />
    </div>
  );
};

export default RoleThreads;
