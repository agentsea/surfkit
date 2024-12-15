import React, { useRef, useEffect, useState } from "react";
import Layout from "../components/Layout";
import { useLocation } from "react-router-dom";
import Task from "../components/Task";
import { getTask } from "../api/Tasks";
import { Typography } from "@material-tailwind/react";

export default function ContainerDesktopPage() {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const agentAddr = queryParams.get("agentAddr") || "http://localhost:9090";
  const taskAddr = queryParams.get("taskAddr") || "http://localhost:9070";
  const vncAddr = queryParams.get("vncAddr") || "http://localhost:3000";
  const taskID = queryParams.get("taskID");
  const token = queryParams.get("authToken");

  const [agentTask, setAgentTask] = useState(null);

  const abortControllerRef = useRef(new AbortController());
  const timeoutRef = useRef(null);
  useEffect(() => {
    const handleStart = async () => {
      console.log("Starting fetch at:", new Date().toISOString());
      const task = await getTask(taskAddr, taskID, token);
      if (!task) {
        return;
      }
      setAgentTask(task);
      console.log("Tasks updated at:", new Date().toISOString());
      console.log(task);
      // Schedule the next call
      timeoutRef.current = setTimeout(handleStart, 2000);
    };

    handleStart(); // Call initially

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current); // Clear the timeout if the component unmounts
      }
    };
  }, [taskAddr, taskID, token]);
  const ref = useRef();

  return (
    <Layout>
      <div className="flex flex-row mt-16 gap-6">
        <div className="min-w-[400px] h-screen">
          {agentTask ? (
            <Task data={agentTask} addr={taskAddr} token={token} />
          ) : (
            <div className="border border-black flex flex-row p-12 items-center justify-center rounded-xl bg-white">
              <Typography variant="h5">No tasks</Typography>
            </div>
          )}
        </div>
        <div className="border border-black flex w-fit h-fit shadow-2xl">
          <iframe
            className="vnc"
            src={`${vncAddr}?autoconnect=1&resize=remote&clipboard_up=true&clipboard_down=true&clipboard_seamless=true&show_control_bar=true`}
            style={{ width: "1280px", height: "858px" }}
          ></iframe>
        </div>
      </div>
    </Layout>
  );
}
