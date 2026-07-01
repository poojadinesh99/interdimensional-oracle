const API_URL = "/api/chat";
const chatHistory = document.getElementById("chat-history");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const tabs = [...document.querySelectorAll(".tab")];
const panes = { chat: document.getElementById("chat-pane"), browse: document.getElementById("browse-pane") };
const searchInput = document.getElementById("browse-search");
const filters = [...document.querySelectorAll(".filter")];
const browseResults = document.getElementById("browse-results");

const browseItems = [
	{ type: "character", title: "Rick Sanchez", summary: "A brilliant, chaotic scientist with a talent for portal travel.", tags: ["genius", "portal gun", "smith family"] },
	{ type: "character", title: "Morty Smith", summary: "Rick's anxious grandson who is pulled into interdimensional adventures.", tags: ["grandson", "anxiety", "family"] },
	{ type: "character", title: "Summer Smith", summary: "Morty's sister and a sharp, resourceful member of the Smith family.", tags: ["sister", "smith family"] },
	{ type: "episode", title: "Pilot", summary: "The first episode that introduces Rick, Morty, and the portal gun.", tags: ["season 1", "intro"] },
	{ type: "episode", title: "The Ricklantis Mixup", summary: "A layered episode centered on an alternate Citadel storyline.", tags: ["citadel", "alternate ricks"] },
	{ type: "episode", title: "Total Rickall", summary: "A fast-paced episode about memory parasites and trust.", tags: ["parasites", "family"] },
	{ type: "location", title: "Citadel of Ricks", summary: "A massive settlement built by Ricks across the multiverse.", tags: ["ricks", "multiverse"] },
	{ type: "location", title: "Earth (C-137)", summary: "The familiar home dimension used throughout the series.", tags: ["earth", "dimension"] },
	{ type: "location", title: "Anatomy Park", summary: "A tiny theme park built inside a human body.", tags: ["park", "body"] },
];

let activeFilter = "all";

function setActiveTab(tab) {
	for (const button of tabs) button.classList.toggle("is-active", button.dataset.tab === tab);
	for (const [name, pane] of Object.entries(panes)) pane.classList.toggle("is-active", name === tab);
}

function getScoreWidth(score) {
	return `${Math.max(0, Math.min(100, Math.round(Number(score || 0) * 100)))}%`;
}

function renderConfidence(score) {
	return `
		<div class="confidence-row">
			<span>Confidence</span>
			<span>${getScoreWidth(score)}</span>
		</div>
		<div class="confidence-bar"><span style="width: ${getScoreWidth(score)}"></span></div>
	`;
}

function renderSourceCards(sources) {
	if (!sources.length) {
		return '<div class="empty-state">No sources returned.</div>';
	}

	return sources
		.map((source) => {
			const kind = String(source.id || "").split(":")[0] || "document";
			return `
				<article class="source-card">
					<div class="source-kicker">📄 ${kind}</div>
					<h4>${source.title || "Untitled"}</h4>
					<p>${source.content ? source.content.slice(0, 140) + (source.content.length > 140 ? "…" : "") : "Retrieved source."}</p>
					<div class="source-score">Score: ${Number(source.score || 0).toFixed(2)}</div>
				</article>
			`;
		})
		.join("");
}

function appendUserMessage(text) {
	const bubble = document.createElement("article");
	bubble.className = "message message-user";
	bubble.innerHTML = `<div class="message-label">You</div><div class="message-text"></div>`;
	bubble.querySelector(".message-text").textContent = text;
	chatHistory.appendChild(bubble);
	chatHistory.scrollTop = chatHistory.scrollHeight;
	return bubble;
}

function appendAssistantPlaceholder() {
	const bubble = document.createElement("article");
	bubble.className = "message message-assistant";
	bubble.innerHTML = `<div class="message-label">Assistant</div><div class="message-text">Thinking…</div>`;
	chatHistory.appendChild(bubble);
	chatHistory.scrollTop = chatHistory.scrollHeight;
	return bubble;
}

function renderAssistantMessage(bubble, data) {
	bubble.innerHTML = `
		<div class="message-label">Assistant</div>
		<div class="message-text">${escapeHtml(data.answer || "No answer returned.")}</div>
		<div class="assistant-meta">${renderConfidence(data.confidence)}</div>
		<div class="source-grid">${renderSourceCards(data.sources || [])}</div>
	`;
	chatHistory.scrollTop = chatHistory.scrollHeight;
}

function escapeHtml(value) {
	return String(value)
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;")
		.replaceAll('"', "&quot;")
		.replaceAll("'", "&#39;");
}

function renderBrowse() {
	const query = searchInput.value.trim().toLowerCase();
	const filtered = browseItems.filter((item) => {
		const matchesType = activeFilter === "all" || item.type === activeFilter;
		const haystack = `${item.title} ${item.summary} ${item.tags.join(" ")}`.toLowerCase();
		return matchesType && haystack.includes(query);
	});

	browseResults.innerHTML = filtered.length
		? filtered
			.map(
				(item) => `
					<article class="browse-card">
						<div class="browse-kind">${item.type}</div>
						<h3>${item.title}</h3>
						<p>${item.summary}</p>
					</article>
				`
			)
			.join("")
		: '<div class="empty-state">No matches found.</div>';
}

tabs.forEach((button) => button.addEventListener("click", () => setActiveTab(button.dataset.tab)));
filters.forEach((button) =>
	button.addEventListener("click", () => {
		activeFilter = button.dataset.filter;
		filters.forEach((item) => item.classList.toggle("is-active", item === button));
		renderBrowse();
	})
);

searchInput.addEventListener("input", renderBrowse);

chatForm.addEventListener("submit", async (event) => {
	event.preventDefault();
	const question = questionInput.value.trim();
	if (!question) return;

	appendUserMessage(question);
	questionInput.value = "";
	const assistantBubble = appendAssistantPlaceholder();

	try {
		const response = await fetch(API_URL, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ question }),
		});
		const data = await response.json();
		if (!response.ok) throw new Error(data.detail || "Something went wrong.");
		renderAssistantMessage(assistantBubble, data);
	} catch (error) {
		renderAssistantMessage(assistantBubble, { answer: error.message, sources: [], confidence: 0 });
	}
});

renderBrowse();
