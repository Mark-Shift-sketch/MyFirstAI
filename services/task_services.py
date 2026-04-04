import json
import re
import time
import datetime
from pathlib import Path
from core.memory import MemoryManager

TASKS_PATH = Path("data/daily_tasks.json")

class TaskService:
    def __init__(self):
        self.daily_planner = self._load_daily_planner()
        self.active_timers = []
        self._last_reminder_tick = ""

    def _load_daily_planner(self):
        default_data = {"tasks": [], "reminders": []}
        if not TASKS_PATH.exists():
            return default_data

        try:
            with TASKS_PATH.open("r", encoding="utf-8") as file_handle:
                loaded = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return default_data

        if not isinstance(loaded, dict):
            return default_data

        tasks = loaded.get("tasks", [])
        reminders = loaded.get("reminders", [])

        if not isinstance(tasks, list):
            tasks = []
        if not isinstance(reminders, list):
            reminders = []

        cleaned_tasks = []
        for item in tasks:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            cleaned_tasks.append(
                {
                    "id": int(item.get("id", 0)) if str(item.get("id", "")).isdigit() else 0,
                    "text": text,
                    "done": bool(item.get("done", False)),
                    "created_at": str(item.get("created_at", "")).strip() or MemoryManager._timestamp_now(),
                    "due": str(item.get("due", "")).strip(),
                }
            )

        cleaned_reminders = []
        for item in reminders:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            cleaned_reminders.append(
                {
                    "id": int(item.get("id", 0)) if str(item.get("id", "")).isdigit() else 0,
                    "text": text,
                    "when": str(item.get("when", "")).strip(),
                    "done": bool(item.get("done", False)),
                    "created_at": str(item.get("created_at", "")).strip() or MemoryManager._timestamp_now(),
                }
            )

        return {"tasks": cleaned_tasks[-200:], "reminders": cleaned_reminders[-200:]}

    def _save_daily_planner(self):
        TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TASKS_PATH.open("w", encoding="utf-8") as file_handle:
            json.dump(self.daily_planner, file_handle, indent=2)

    @staticmethod
    def _next_item_id(items):
        highest = 0
        for item in items:
            try:
                highest = max(highest, int(item.get("id", 0)))
            except (TypeError, ValueError):
                continue
        return highest + 1

    @staticmethod
    def _extract_first_number(text):
        match = re.search(r"\b(\d+)\b", text)
        if not match:
            return None
        return int(match.group(1))

    def _task_overview(self, include_completed=False, limit=6):
        tasks = self.daily_planner.get("tasks", [])
        if not include_completed:
            tasks = [task for task in tasks if not task.get("done")]

        if not tasks:
            return "No active tasks. You are clear for now."

        lines = []
        for task in tasks[:limit]:
            due = task.get("due", "")
            suffix = f" (due {due})" if due else ""
            lines.append(f"task {task.get('id')}: {task.get('text')}{suffix}")

        if len(tasks) > limit:
            lines.append(f"plus {len(tasks) - limit} more")

        return "; ".join(lines) + "."

    def _reminder_overview(self, include_completed=False, limit=5):
        reminders = self.daily_planner.get("reminders", [])
        if not include_completed:
            reminders = [item for item in reminders if not item.get("done")]

        if not reminders:
            return "No active reminders."

        lines = []
        for reminder in reminders[:limit]:
            when = reminder.get("when", "")
            suffix = f" at {when}" if when else ""
            lines.append(f"reminder {reminder.get('id')}: {reminder.get('text')}{suffix}")

        if len(reminders) > limit:
            lines.append(f"plus {len(reminders) - limit} more")

        return "; ".join(lines) + "."

    def _add_task(self, text, due=""):
        tasks = self.daily_planner.setdefault("tasks", [])
        task_id = self._next_item_id(tasks)
        tasks.append(
            {
                "id": task_id,
                "text": text,
                "done": False,
                "created_at": MemoryManager._timestamp_now(),
                "due": due,
            }
        )
        self.daily_planner["tasks"] = tasks[-200:]
        self._save_daily_planner()

        if due:
            return f"Task {task_id} added: {text}, due {due}."
        return f"Task {task_id} added: {text}."

    def _set_task_done(self, task_id):
        for task in self.daily_planner.get("tasks", []):
            if int(task.get("id", 0)) == task_id:
                if task.get("done"):
                    return f"Task {task_id} is already completed."
                task["done"] = True
                self._save_daily_planner()
                return f"Task {task_id} completed."
        return f"I could not find task {task_id}."

    def _remove_task(self, task_id):
        tasks = self.daily_planner.get("tasks", [])
        kept = [task for task in tasks if int(task.get("id", 0)) != task_id]
        if len(kept) == len(tasks):
            return f"I could not find task {task_id}."
        self.daily_planner["tasks"] = kept
        self._save_daily_planner()
        return f"Task {task_id} removed."

    def _clear_completed_tasks(self):
        tasks = self.daily_planner.get("tasks", [])
        active = [task for task in tasks if not task.get("done")]
        removed = len(tasks) - len(active)
        self.daily_planner["tasks"] = active
        self._save_daily_planner()
        return f"Cleared {removed} completed tasks."

    def _add_reminder(self, text, when=""):
        reminders = self.daily_planner.setdefault("reminders", [])
        reminder_id = self._next_item_id(reminders)
        reminders.append(
            {
                "id": reminder_id,
                "text": text,
                "when": when,
                "done": False,
                "created_at": MemoryManager._timestamp_now(),
            }
        )
        self.daily_planner["reminders"] = reminders[-200:]
        self._save_daily_planner()

        if when:
            return f"Reminder {reminder_id} added for {when}: {text}."
        return f"Reminder {reminder_id} added: {text}."

    def _set_reminder_done(self, reminder_id):
        for reminder in self.daily_planner.get("reminders", []):
            if int(reminder.get("id", 0)) == reminder_id:
                if reminder.get("done"):
                    return f"Reminder {reminder_id} is already completed."
                reminder["done"] = True
                self._save_daily_planner()
                return f"Reminder {reminder_id} completed."
        return f"I could not find reminder {reminder_id}."

    def _daily_brief(self):
        now = datetime.datetime.now()
        today = now.strftime("%A, %B %d")
        current_time = now.strftime("%I:%M %p")

        pending_tasks = [task for task in self.daily_planner.get("tasks", []) if not task.get("done")]
        active_reminders = [
            reminder for reminder in self.daily_planner.get("reminders", []) if not reminder.get("done")
        ]

        if pending_tasks:
            top_task = pending_tasks[0].get("text", "")
            tasks_line = f"{len(pending_tasks)} active tasks. Next priority: {top_task}."
        else:
            tasks_line = "No active tasks. You can add one by saying add task followed by the task."

        reminders_line = (
            f"{len(active_reminders)} active reminders."
            if active_reminders
            else "No active reminders."
        )

        return f"Daily brief for {today}, {current_time}. {tasks_line} {reminders_line}"

    @staticmethod
    def _format_duration(seconds):
        seconds = int(max(0, seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        parts = []
        if hours:
            parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
        if minutes:
            parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
        if secs or not parts:
            parts.append(f"{secs} second" + ("s" if secs != 1 else ""))
        return " ".join(parts)

    @staticmethod
    def _parse_timer_duration_seconds(text):
        units = re.findall(
            r"(\d+)\s*(hours?|hrs?|hr|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)",
            text.lower(),
        )
        if not units:
            return None

        total = 0
        for amount_raw, unit in units:
            amount = int(amount_raw)
            if amount < 0:
                return None
            if unit.startswith(("hour", "hr", "h")):
                total += amount * 3600
            elif unit.startswith(("minute", "min", "m")):
                total += amount * 60
            else:
                total += amount

        if total <= 0 or total > 24 * 3600:
            return None
        return total

    @staticmethod
    def _parse_clock_time(value):
        probe = value.strip().lower().replace(".", "")
        for fmt in ("%H:%M", "%I:%M %p", "%I %p"):
            try:
                parsed = datetime.datetime.strptime(probe.upper(), fmt)
                return parsed.hour, parsed.minute
            except ValueError:
                continue
        return None

    def _set_timer(self, seconds, label="timer"):
        timer_id = len(self.active_timers) + 1
        self.active_timers.append(
            {
                "id": timer_id,
                "label": label,
                "due": time.monotonic() + float(seconds),
                "duration": int(seconds),
            }
        )
        return f"Timer {timer_id} set for {self._format_duration(seconds)}."

    def _list_timers(self):
        if not self.active_timers:
            return "No active timers."

        now = time.monotonic()
        parts = []
        for timer in self.active_timers[:5]:
            remaining = max(0, int(timer["due"] - now))
            parts.append(
                f"timer {timer['id']} {timer['label']} with {self._format_duration(remaining)} remaining"
            )
        if len(self.active_timers) > 5:
            parts.append(f"plus {len(self.active_timers) - 5} more")
        return "; ".join(parts) + "."

    def _cancel_timer(self, timer_id=None):
        if not self.active_timers:
            return "No active timers to cancel."

        if timer_id is None:
            self.active_timers.clear()
            return "All active timers canceled."

        kept = [timer for timer in self.active_timers if int(timer.get("id", 0)) != timer_id]
        if len(kept) == len(self.active_timers):
            return f"I could not find timer {timer_id}."

        self.active_timers = kept
        return f"Timer {timer_id} canceled."

    def _check_due_timers(self):
        if not self.active_timers:
            return []

        now = time.monotonic()
        due = [timer for timer in self.active_timers if timer["due"] <= now]
        if not due:
            return []

        due_ids = {timer["id"] for timer in due}
        self.active_timers = [timer for timer in self.active_timers if timer["id"] not in due_ids]
        return [
            f"Timer {timer['id']} complete for {timer.get('label', 'timer')}."
            for timer in due
        ]

    def _check_due_reminders(self):
        now = datetime.datetime.now()
        tick_key = now.strftime("%Y-%m-%d %H:%M")
        if tick_key == self._last_reminder_tick:
            return []
        self._last_reminder_tick = tick_key

        fired = []
        for reminder in self.daily_planner.get("reminders", []):
            if reminder.get("done"):
                continue
            when_text = str(reminder.get("when", "")).strip()
            if not when_text:
                continue

            parsed = self._parse_clock_time(when_text)
            if not parsed:
                continue

            hour, minute = parsed
            if now.hour == hour and now.minute == minute:
                reminder["done"] = True
                fired.append(f"Reminder {reminder.get('id')}: {reminder.get('text')}")

        if fired:
            self._save_daily_planner()
        return fired

    def _bulk_add_tasks(self, task_texts):
        task_texts = [text.strip() for text in task_texts if text and text.strip()]
        if not task_texts:
            return []

        tasks = self.daily_planner.setdefault("tasks", [])
        next_id = self._next_item_id(tasks)
        added_ids = []
        for text in task_texts:
            tasks.append(
                {
                    "id": next_id,
                    "text": text,
                    "done": False,
                    "created_at": MemoryManager._timestamp_now(),
                    "due": "",
                }
            )
            added_ids.append(next_id)
            next_id += 1

        self.daily_planner["tasks"] = tasks[-200:]
        self._save_daily_planner()
        return added_ids

    def _generate_goal_steps(self, goal):
        goal_lower = goal.lower()
        if any(token in goal_lower for token in ("study", "learn", "exam", "course")):
            return [
                f"Define the exact learning target for {goal}",
                "Split the topic into four focused subtopics",
                "Create a daily 45-minute study block with no distractions",
                "Practice with exercises and track weak areas",
                "Review progress at end of day and adjust next session",
            ]
        if any(token in goal_lower for token in ("project", "build", "app", "system")):
            return [
                f"Write a one-sentence objective for {goal}",
                "Break implementation into milestone tasks",
                "Deliver a minimal working version first",
                "Test core flows and fix blockers",
                "Polish, document, and finalize",
            ]
        return [
            f"Clarify success criteria for {goal}",
            "Break the goal into five executable actions",
            "Schedule the first action in your next focus block",
            "Track completion and remove blockers daily",
            "Review results and set next iteration",
        ]

    def _build_goal_plan(self, goal, conversation_state, auto_add_tasks=False):
        steps = self._generate_goal_steps(goal)
        conversation_state["last_plan_goal"] = goal
        conversation_state["last_plan_steps"] = steps

        if auto_add_tasks:
            added_ids = self._bulk_add_tasks(steps)
            listed = ", ".join(str(i) for i in added_ids)
            return (
                f"Plan created for {goal}. I added {len(added_ids)} tasks to your list "
                f"with IDs {listed}."
            )

        numbered = " ".join(f"step {index + 1}: {step}." for index, step in enumerate(steps))
        return f"Strategic plan for {goal}: {numbered}"
