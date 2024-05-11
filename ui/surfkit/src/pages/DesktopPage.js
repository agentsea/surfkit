import React, { useRef, useEffect, useState } from "react";
import { VncScreen } from "react-vnc";
import Layout from "../components/Layout";
import { useLocation } from "react-router-dom";
import Task from "../components/Task";
import { getTasks } from "../api/Tasks";
import { Typography } from "@material-tailwind/react";

export default function DesktopPage() {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const agentAddr = queryParams.get("agentAddr") || "http://localhost:9090";
  const vncAddr = queryParams.get("vncAddr") || "ws://localhost:6080";

  const [agentTasks, setAgentTasks] = useState(null);

  useEffect(() => {
    let abortController = new AbortController();
    const handleStart = async () => {
      const tasks = await getTasks(agentAddr, abortController.signal);
      if (!tasks) {
        return;
      }
      setAgentTasks(tasks);
    };
    handleStart();

    const intervalId = setInterval(() => {
      abortController.abort(); // Cancel the previous request
      abortController = new AbortController(); // Create a new controller for the new request
      handleStart();
    }, 2000);

    return () => {
      clearInterval(intervalId);
      abortController.abort(); // Ensure fetch is also cancelled when the component unmounts
    };
  }, [agentAddr]);
  const ref = useRef();

  return (
    <Layout>
      <div className="flex flex-row mt-16 gap-6">
        <div className="min-w-[400px] h-screen">
          {agentTasks ? (
            <Task data={agentTasks[0]} addr={agentAddr} />
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
