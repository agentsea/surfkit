export async function getTasks(addr, token) {
  const url = new URL(`/v1/tasks`, addr);
  try {
    const headers = {
      "Content-Type": "application/json",
    };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const resp = await fetch(url, {
      method: "GET",
      cache: "no-cache",
      headers: headers,
      redirect: "follow",
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

export async function getTask(addr, id, token) {
  const url = new URL(`/v1/tasks/${id}`, addr);
  try {
    const headers = {
      "Content-Type": "application/json",
    };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const resp = await fetch(url, {
      method: "GET",
      cache: "no-cache",
      headers: headers,
      redirect: "follow",
    });
    if (!resp.ok) {
      throw new Error("HTTP status " + resp.status);
    }
    const data = await resp.json();
    return data;
  } catch (error) {
    console.error("Failed to get task", error);
  }
}

export async function updateTask(addr, id, bodyData, token) {
  console.log("updating task with id: ", id);
  console.log("bodyData: ", bodyData);
  console.log("addr: ", addr);
  const url = new URL(`/v1/tasks/${id}`, addr);
  console.log("updating tasks with URL: ", url);

  try {
    const headers = {
      "Content-Type": "application/json",
    };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const resp = await fetch(url, {
      method: "PUT",
      cache: "no-cache",
      headers: headers,
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
