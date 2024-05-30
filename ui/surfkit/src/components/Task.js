import { Typography, Button, Chip } from "@material-tailwind/react";
import { useRef, useEffect, useState } from "react";
import {
  ClipboardIcon,
  XCircleIcon,
  CheckCircleIcon,
} from "@heroicons/react/24/outline";
import { motion } from "framer-motion";
import { updateTask } from "../api/Tasks";
import RoleThreads from "./RoleThreads";

export default function Task({ data, addr }) {
  const endOfMessagesRef = useRef(null);
  const prevMessagesLength = useRef(data?.thread?.messages.length || 0);
  const [message, setMessage] = useState(null);
  const [thread, setThread] = useState("feed");
  const messageContainerRef = useRef(null);
  const [isUserAtBottom, setIsUserAtBottom] = useState(true);
  const [activeMessages, setActiveMessages] = useState([]);

  const threads = data.threads || [];

  const handleCancelTask = () => {
    console.log("cancelling task...");
    updateTask(addr, data.id, { status: "canceling" });
  };

  const handleFailTask = () => {
    console.log("failing task...");
    updateTask(addr, data.id, { status: "failed" });
  };

  const handleCompleteTask = (stat) => {
    console.log("completing task...");
    updateTask(addr, data.id, { status: "completed" });
  };

  const handleKeyDown = async (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      console.log("message:");
      console.log(message);

      var msgData = {
        role: "user",
        msg: message,
      };
      console.log("data: ");
      console.log(data);
      // postTaskMessage(data.id, msgData, token);
      setMessage("");
    }
  };

  useEffect(() => {
    console.log("thread changed: ", thread);
  }, [thread]);

  const getActiveThreadMessages = () => {
    const threadMessages =
      threads.find((thrd) => thrd.name === thread)?.messages || [];
    return threadMessages;
  };

  useEffect(() => {
    // Function to handle auto-scrolling logic
    const handleAutoScroll = () => {
      if (isUserAtBottom && messageContainerRef.current) {
        messageContainerRef.current.scrollTop =
          messageContainerRef.current.scrollHeight;
      }
    };

    // Call the auto-scroll function whenever messages are updated
    handleAutoScroll();
  }, [activeMessages, isUserAtBottom]); // Depend on the messages count

  useEffect(() => {
    console.log(`Thread changed to: ${thread}`);
    // Log current thread messages for debugging
    console.log(
      `Current thread messages for ${thread}:`,
      getActiveThreadMessages()
    );
    // Add any additional logic here if you need to fetch new messages when the thread changes
  }, [thread]);

  useEffect(() => {
    const handleScroll = () => {
      if (!messageContainerRef.current) return;

      const { scrollTop, clientHeight, scrollHeight } =
        messageContainerRef.current;
      // Set isUserAtBottom based on whether the user is scrolled to within 10 pixels of the bottom
      setIsUserAtBottom(scrollHeight - scrollTop <= clientHeight + 10);
    };

    // Add scroll event listener
    const currentContainer = messageContainerRef.current;
    currentContainer?.addEventListener("scroll", handleScroll);

    return () => {
      // Clean up event listener
      currentContainer?.removeEventListener("scroll", handleScroll);
    };
  }, []);

  useEffect(() => {
    setActiveMessages(getActiveThreadMessages());
  }, [thread, data.thread?.messages]);

  const getChipColor = (status) => {
    switch (status) {
      case "completed":
        return "green";
      case "error":
        return "red";
      case "failed":
        return "red";
      case "review":
        return "purple";
      case "canceled":
        return "gray";
      case "canceling":
        return "yellow";
      default:
        return "blue";
    }
  };

  useEffect(() => {
    // Check if a new message was added by comparing the current length to the previous one
    if (data.thread?.messages.length > prevMessagesLength.current) {
      // Use the scrollTo method with top equal to the element's offsetTop. This way, it scrolls within its container.
      const scrollContainer = endOfMessagesRef.current?.parentNode;
      if (scrollContainer && endOfMessagesRef.current) {
        scrollContainer.scrollTo({
          top: endOfMessagesRef.current.offsetTop,
          behavior: "smooth",
        });
      }
    }
    // Update the previous length for the next render
    prevMessagesLength.current = data.thread?.messages.length;
  }, [data.thread?.messages.length]);
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="flex flex-col gap-4 border w-full border-black-400 bg-white shadow-lg rounded-md p-2 h-[720px]"
    >
      <div className="flex flex-row">
        <ClipboardIcon className="w-4 h-4 mt-1 mr-1"></ClipboardIcon>
        <Typography color="blue-gray">Task</Typography>
        {data.status &&
          data.status !== "completed" &&
          data.status !== "failed" &&
          data.status !== "review" &&
          data.status !== "cancelled" && (
            <div className="w-full justify-end flex flex-row gap-2">
              <Button
                variant="text"
                onClick={handleCancelTask}
                size="sm"
                color="red"
                className="normal-case flex items-center gap-3"
              >
                <XCircleIcon className="w-4 h-4" />
                Cancel
              </Button>
              <Button
                variant="text"
                onClick={handleCompleteTask}
                color="green"
                className="normal-case flex items-center gap-3"
              >
                <CheckCircleIcon className="w-4 h-4" />
                Complete
              </Button>
            </div>
          )}
        {data.status && data.status === "review" && (
          <div className="w-full justify-end flex flex-row gap-2">
            <Button
              variant="text"
              onClick={handleFailTask}
              size="sm"
              color="red"
              className="normal-case flex items-center gap-3"
            >
              <XCircleIcon className="w-4 h-4" />
              Fail
            </Button>
            <Button
              variant="text"
              onClick={handleCompleteTask}
              color="green"
              className="normal-case flex items-center gap-3"
            >
              <CheckCircleIcon className="w-4 h-4" />
              Complete
            </Button>
          </div>
        )}
      </div>
      <div className="gap-2">
        <div className="flex flex-row">
          <Typography variant="paragraph" className="font-semibold mr-2">
            Description
          </Typography>
          {/* TODO: this should be handled better */}
          <div className="max-h-[75px] overflow-y-scroll">
            <Typography variant="paragraph">{data.description}</Typography>
          </div>
        </div>
        <div className="flex flex-row mt-2">
          <Typography variant="paragraph" className="font-semibold mr-2">
            Status
          </Typography>
          <Chip
            color={getChipColor(data.status)}
            size="sm"
            className="rounded-full"
            variant="ghost"
            value={data.status}
          />
        </div>
        <div className="flex flex-col h-full overflow-y-scroll">
          <RoleThreads threads={threads} />
        </div>
      </div>
      {/* <div
        className="relative max-h-80 overflow-y-scroll border border-black-400 p-2 rounded-xl gap-2 flex flex-col"
        ref={messageContainerRef}
      >
        {getActiveThreadMessages().map((msg, index) => (
          <div className="flex flex-row gap-2" key={index}>
            <Avatar
              className="w-4 h-4 mt-2 mr-2"
              src="https://storage.googleapis.com/guisurfer-assets/surf_dino2.webp"
            />
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="border border-black-400 p-3 bg-neutral-50 rounded-xl flex flex-row"
            >
              <Typography variant="small" className="font-mono">
                {msg.text}
              </Typography>
            </motion.div>
          </div>
        ))}
        <div ref={endOfMessagesRef} />
      </div> */}
      {/* {data.status &&
        data.status !== "completed" &&
        data.status !== "cancelled" &&
        data.status !== "failed" && (
          <div className="w-full pb-4 px-4">
            <Input
              variant="standard"
              label="Send a message"
              icon={<PaperAirplaneIcon />}
              onKeyDown={handleKeyDown}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
            />
          </div>
        )} */}
      {data.output && data.output !== "" && (
        <div className="flex flex-row">
          <Typography variant="paragraph" className="font-semibold mr-2">
            Result
          </Typography>
          <Typography variant="paragraph">{data.output}</Typography>
        </div>
      )}
      {data.error && data.error !== "" && (
        <div className="flex flex-row overflow-y-scroll">
          <Typography
            variant="paragraph"
            className="font-semibold mr-2"
            color="red"
          >
            Error
          </Typography>
          <Typography variant="paragraph">{data.error}</Typography>
        </div>
      )}
    </motion.div>
  );
}
