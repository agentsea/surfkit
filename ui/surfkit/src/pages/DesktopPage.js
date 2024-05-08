import React, { useRef, useEffect, useState } from "react";
import { VncScreen } from "react-vnc";
import Layout from "../components/Layout";
import { useLocation } from "react-router-dom";
import Task from "../components/Task";
import { getTasks } from "../api/Tasks";

export default function DesktopPage() {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const agentAddr = queryParams.get("agentAddr") || "http://localhost:9090";
  const vncAddr = queryParams.get("vncAddr") || "ws://localhost:6080";

  const [agentTasks, setAgentTasks] = useState(null);

  useEffect(() => {
    const handleStart = async () => {
      var tasks = await getTasks(agentAddr);
      if (!tasks) {
        return;
      }
      setAgentTasks(tasks);
      console.log("got agent tasks: ", tasks);
    };
    handleStart();
    // Then set the interval
    const intervalId = setInterval(async () => {
      var tasks_ = await getTasks(agentAddr);
      if (!tasks_) {
        return;
      }
      setAgentTasks(tasks_);
      console.log("got agent tasks: ", tasks_);
    }, 2000);

    // Clear the interval when the component is unmounted
    return () => clearInterval(intervalId);
  }, [agentAddr]);

  const ref = useRef();

  // const agentdAddr = "http://localhost:8000";

  return (
    <Layout>
      <div className="flex flex-row mt-16 gap-6">
        {agentTasks && <Task data={agentTasks[0]} />}
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
