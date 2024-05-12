import React, { useState, useEffect, useRef } from "react";
import { Typography, Avatar, Dialog } from "@material-tailwind/react";
import { motion } from "framer-motion";

const RoleThread = ({ data }) => {
  console.log("thread data: ");
  console.log(data);
  const messages = data?.messages || [];
  const roleMapping = data?.role_mapping || {};
  console.log("!role mapping: ", roleMapping);
  const [modalOpen, setModalOpen] = useState(false);
  const [currentImage, setCurrentImage] = useState("");
  const endOfMessagesRef = useRef(null);

  // Function to handle image click
  const handleImageClick = (image) => {
    setCurrentImage(image);
    setModalOpen(true);
  };

  useEffect(() => {
    // Automatically scroll to the end of messages when new messages are added
    const scrollContainer = endOfMessagesRef.current?.parentNode;
    if (scrollContainer && endOfMessagesRef.current) {
      scrollContainer.scrollTo({
        top: endOfMessagesRef.current.offsetTop,
        behavior: "smooth",
      });
    }
  }, [messages.length]);

  return (
    <div className="flex flex-col gap-2 relative overflow-y-scroll border border-black-400 max-h-[480px]">
      {messages.map((msg, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="border border-black-400 p-3 bg-neutral-50 rounded-xl flex flex-col mt-1"
        >
          <div className="flex flex-row items-center gap-2">
            {/* Render the avatar with fallback */}
            <Avatar
              src={
                roleMapping[msg.role]?.icon ||
                "https://cdn-icons-png.flaticon.com/512/1053/1053244.png"
              }
              alt="Role Avatar"
              className="w-5 h-5"
            />
            {/* Role Name or Fallback */}
            <Typography variant="small" className="font-semibold">
              {roleMapping[msg.role]?.user_name || msg.role}
            </Typography>
          </div>
          <Typography variant="small" className="font-mono mb-2 mt-2">
            {msg.text}
          </Typography>
          {/* If there are images, render them */}
          {msg.images &&
            msg.images.map((image, imgIndex) => (
              <img
                key={imgIndex}
                src={image}
                alt={`Message Attachment ${imgIndex}`}
                className="w-16 h-16 cursor-pointer mt-2"
                onClick={() => handleImageClick(image)}
              />
            ))}
        </motion.div>
      ))}
      {/* Invisible marker for scrolling */}
      <div ref={endOfMessagesRef} />
      {/* Modal for displaying the full-size image */}
      <Dialog
        open={modalOpen}
        handler={() => setModalOpen(false)}
        className="max-w-full max-h-full overflow-y-auto"
      >
        <img
          src={currentImage}
          alt="Modal Content"
          className="max-w-full max-h-[90vh] object-contain"
        />
      </Dialog>
    </div>
  );
};

export default RoleThread;
