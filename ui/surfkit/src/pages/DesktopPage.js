import React, { useRef, useEffect, useState } from "react";
import { VncScreen } from "react-vnc";
import Layout from "../components/Layout";
import { useLocation } from "react-router-dom";
import Task from "../components/Task";
import { getTask } from "../api/Tasks";
import { Typography } from "@material-tailwind/react";

export default function DesktopPage() {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const agentAddr = queryParams.get("agentAddr") || "http://localhost:9090";
  const taskAddr = queryParams.get("taskAddr") || "http://localhost:9070";
  const vncAddr = queryParams.get("vncAddr") || "ws://localhost:6080";
  const taskID = queryParams.get("taskID");

  const [agentTask, setAgentTask] = useState(null);

  const abortControllerRef = useRef(new AbortController());
  const timeoutRef = useRef(null);
  useEffect(() => {
    const handleStart = async () => {
      console.log("Starting fetch at:", new Date().toISOString());
      const task = await getTask(taskAddr, taskID);
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
  }, [taskAddr, taskID]);
  const ref = useRef();

  return (
    <Layout>
      <div className="flex flex-row mt-16 gap-6">
        <div className="min-w-[400px] h-screen">
          {agentTask ? (
            <Task data={agentTask} addr={taskAddr} />
          ) : (
            <div className="border border-black flex flex-row p-12 items-center justify-center rounded-xl bg-white">
              <Typography variant="h5">No tasks</Typography>
            </div>
          )}
        </div>
        <div className="border border-black flex w-fit h-fit shadow-2xl">
          <VncScreen
            url={vncAddr}
            scaleViewport
            background="#000000"
            style={{
              width: "900px",
              height: "720px",
            }}
            rfbOptions={{
              credentials: { password: "agentsea123" }, // this is fine, we tunnel everything through ssh, some vnc viewers require a password
            }}
            ref={ref}
          />
        </div>
      </div>
    </Layout>
  );
}
