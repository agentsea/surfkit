export async function getTasks(addr, signal) {
  const url = new URL(`/v1/tasks`, addr);
  try {
    const resp = await fetch(url, {
      method: "GET",
      cache: "no-cache",
      headers: {
        "Content-Type": "application/json",
      },
      redirect: "follow",
      signal,
    });
    if (!resp.ok) {
      throw new Error("HTTP status " + resp.status);
    }
    const data = await resp.json();
    return data.tasks;
  } catch (error) {
    console.error("Failed to list tasks", error);
  }
}

export async function updateTask(addr, id, bodyData) {
  console.log("updating task with id: ", id);
  console.log("bodyData: ", bodyData);
  console.log("addr: ", addr);
  const url = new URL(`/v1/tasks/${id}`, addr);
  console.log("updating tasks with URL: ", url);

  try {
    const resp = await fetch(url, {
      method: "PUT",
      cache: "no-cache",
      headers: {
        "Content-Type": "application/json",
      },
      redirect: "follow",
      body: JSON.stringify(bodyData),
    });
    if (!resp.ok) {
      throw new Error("HTTP status " + resp.status);
    }
    console.log("Updated task successfully");
    const data = await resp.json();
    console.log("Got update tasks response: ", data);
    return data.tasks;
  } catch (error) {
    console.error("Failed to update task", error);
  }
}
