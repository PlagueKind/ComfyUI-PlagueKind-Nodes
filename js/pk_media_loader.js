import { app } from "../../scripts/app.js";

// PK Media Loader — tabbed slot UI.
//
// The node has 12 file-picker combos (image_1..9, video_1..3) plus 3
// per-video audio toggles (video_1_audio..3). This extension turns it into
// a tabbed interface:
//   - a tab bar across the top (I1..I9, V1..V3, ALL) — 13 tabs
//   - only the active slot's combo + upload button + preview are shown
//     (video tabs also show their audio toggle)
//   - image slots preview the selected file; video slots get a <video> player
//   - ALL restores the normal view with every slot + toggle
//
// Tab state lives in node.flags (persisted with the workflow, never sent in
// the prompt). The extra widgets (tab bar, upload button, preview) are
// non-serialized. Uploads go straight to /upload/image (the same endpoint the
// core file picker uses).

const SLOTS = [
	"image_1", "image_2", "image_3", "image_4", "image_5",
	"image_6", "image_7", "image_8", "image_9",
	"video_1", "video_2", "video_3",
];
// Bundle channels (15): images 1-9 (input_1..9), video frames 1-3
// (input_10..12), video audio 1-3 (input_13..15)
const BUNDLE_CHANNELS = 15;

// Per-video audio toggles (bools, default on) - shown with their video tab
const AUDIO_TOGGLES = ["video_1_audio", "video_2_audio", "video_3_audio"];

const ALL = "all";
const TAB_FLAG = "pk_media_slot";
const TAB_BAR_NAME = "$$pk_media_tabs";
const UPLOAD_NAME = "$$pk_media_upload";
const PREVIEW_NAME = "$$pk_media_preview";
const CANVAS_PREVIEW_NAME = "$$canvas-image-preview";
const NONE = "(none)";

const IMAGE_ACCEPT = "image/png,image/jpeg,image/webp,image/gif,image/bmp,image/tga";
const VIDEO_ACCEPT = "video/mp4,video/quicktime,video/x-m4v,video/webm,.mp4,.mov,.m4v,.webm";

function isVideoSlot(name) {
	return name.startsWith("video_");
}

function comboLabel(name) {
	const m = name.match(/^(image|video)_(\d+)$/);
	if (!m) return name;
	return (m[1] === "image" ? "I" : "V") + m[2];
}

function findWidget(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

function removeWidget(node, name) {
	const w = findWidget(node, name);
	if (!w) return false;
	w.onRemove?.();
	node.widgets.splice(node.widgets.indexOf(w), 1);
	return true;
}

function slotFileUrl(file) {
	// this frontend serves media through the API view endpoint
	return (
		"/api/view?type=input&filename=" + encodeURIComponent(file) +
		"&subfolder=&rand=" + Math.random()
	);
}

function dirty(node) {
	node.setDirtyCanvas?.(true, true);
	node.graph?.setDirtyCanvas?.(true, true);
}

/**
 * Grow the node to fit its current (visible) content — never shrink it.
 * A size the user set by hand (dragging the resize handle) is treated as
 * pinned and preserved across tab switches; content that no longer fits
 * still forces a grow, exactly like the core layout does.
 */
function fitNode(node) {
	const doFit = () => {
		try {
			const [cw, ch] = node.computeSize();
			const w = Math.max(node.size[0], cw);
			const h = Math.max(node.size[1], ch);
			if (w !== node.size[0] || h !== node.size[1]) {
				node.setSize([w, h]);
			}
		} catch (e) {
			// node not on a graph yet etc. — ignore
		}
		dirty(node);
	};
	doFit();
	// DOM preview heights settle after a frame / after images load
	requestAnimationFrame(doFit);
	setTimeout(doFit, 250);
}

/**
 * Preview strategy:
 *  - image tabs: node.imgs = [the active slot's image] — the frontend's own
 *    canvas preview renders a single image (one entry = no thumbnail strip).
 *    We cooperate with it instead of fighting it.
 *  - video tabs: node.imgs cleared + a DOM <video> widget (the canvas system
 *    can't play mp4s).
 */
function getImageEl(node, file) {
	if (!node.__pkImgCache) node.__pkImgCache = new Map();
	let img = node.__pkImgCache.get(file);
	if (!img) {
		img = new Image();
		img.src = slotFileUrl(file);
		img.alt = file;
		node.__pkImgCache.set(file, img);
	}
	return img;
}

function showVideoWidget(node, slot, file) {
	const existing = findWidget(node, PREVIEW_NAME);
	if (existing && existing.__pkFile === file) return; // already showing it
	if (existing) removeWidget(node, PREVIEW_NAME);

	const div = document.createElement("div");
	div.style.cssText =
		"display:flex;justify-content:center;align-items:center;padding:4px 0 2px 0;overflow:hidden;";
	const v = document.createElement("video");
	v.src = slotFileUrl(file);
	v.controls = true;
	v.preload = "metadata";
	v.style.cssText =
		"max-width:100%;max-height:100%;border-radius:4px;background:#000;";

	// track the video's real aspect ratio so the widget's layout height
	// matches the rendered player (mirrors the core video-preview widget).
	// The player fills whatever room the node has (pinned sizes included) —
	// re-run on metadata load and from the guard tick so manual resizes
	// re-fit it.
	let h = 120;
	const applyDims = () => {
		const vw = v.videoWidth, vh = v.videoHeight;
		if (vw && vh) {
			const maxW = Math.max(200, (node.size?.[0] ?? 330) - 20);
			const capH = Math.max(160, (node.size?.[1] ?? 240) - 150);
			h = Math.min(capH, Math.max(80, Math.round((maxW * vh) / vw)));
		}
		if (div.style.height === h + "px") return; // no change — no redraw
		div.style.height = h + "px";
		const wgt = findWidget(node, PREVIEW_NAME);
		if (wgt) wgt.__pkH = h;
		dirty(node);
	};
	v.onloadedmetadata = applyDims;
	v.onerror = applyDims;
	requestAnimationFrame(applyDims);

	div.appendChild(v);
	const w = node.addDOMWidget(PREVIEW_NAME, "string", div, {
		serialize: false,
		surfaces: { canvas: "shown", vueNode: "never", panel: "never" },
	});
	w.__pkFile = file;
	w.__pkH = h;
	w.__pkApplyDims = applyDims; // guard tick re-fits after manual resizes
	w.computeLayoutSize = () => ({
		minHeight: w.__pkH ?? 120,
		minWidth: 220, // player width follows the node (div is 100%); don't
		              // feed the node width back in — it loops and grows
	});
	fitNode(node);
}

/**
 * Make the node's preview match the active slot. Idempotent — safe to call
 * from the periodic guard.
 */
function syncPreview(node, slot) {
	const combo = findWidget(node, slot);
	const value = combo?.value;

	if (isVideoSlot(slot)) {
		// video: DOM player, canvas preview must stay empty
		if (node.imgs) {
			node.imgs = undefined;
			node.imageIndex = null;
		}
		if (value && value !== NONE) {
			showVideoWidget(node, slot, value);
		} else if (findWidget(node, PREVIEW_NAME)) {
			removeWidget(node, PREVIEW_NAME);
		}
		return;
	}

	// image slot: drop any DOM video widget, drive the canvas preview
	if (findWidget(node, PREVIEW_NAME)) removeWidget(node, PREVIEW_NAME);
	if (!value || value === NONE) {
		if (node.imgs) {
			node.imgs = undefined;
			node.imageIndex = null;
		}
		return;
	}
	const img = getImageEl(node, value);
	if (!node.imgs || node.imgs[0] !== img) {
		node.imgs = [img];
		node.imageIndex = 0;
		dirty(node);
	}
}

/**
 * Refresh the file lists of all slot combos from the server (after uploads).
 */
async function refreshFileLists(node) {
	try {
		const resp = await fetch("/object_info/PK_MediaLoader");
		if (!resp.ok) return;
		const info = await resp.json();
		const req = info?.["PK_MediaLoader"]?.input?.required ?? {};
		for (const name of SLOTS) {
			const w = findWidget(node, name);
			const entry = req[name];
			const list = Array.isArray(entry) ? entry[0] : null;
			if (w && Array.isArray(list)) {
				w._rawOptions.values = list;
				try { w.options.values = list; } catch (e) {}
			}
		}
	} catch (e) {
		// best effort only
	}
}

/**
 * Upload a file picked from disk into the input folder and point the active
 * slot at it. Mirrors what the core file picker does.
 */
async function handleFilePicked(node, slot, file) {
	try {
		const fd = new FormData();
		fd.append("image", file);
		fd.append("subfolder", "");
		fd.append("type", "input");
		const resp = await fetch("/upload/image", { method: "POST", body: fd });
		if (!resp.ok) throw new Error("HTTP " + resp.status);
		const data = await resp.json();
		const name = data?.name;
		if (!name) throw new Error("no filename in response");
		await refreshFileLists(node);
		const combo = findWidget(node, slot);
		if (combo) {
			combo.value = name;
			combo.callback?.(name);
		}
	} catch (e) {
		console.error("[PK Media Loader] upload failed:", e);
	}
}

function createUploadButton(node) {
	if (findWidget(node, UPLOAD_NAME)) return;
	const input = document.createElement("input");
	input.type = "file";
	input.style.display = "none";
	input.multiple = false;
	document.body.appendChild(input);
	input.onchange = async () => {
		const file = input.files?.[0];
		input.value = "";
		if (!file) return;
		const slot = node.flags?.[TAB_FLAG] ?? "image_1";
		if (slot === ALL || !SLOTS.includes(slot)) return;
		await handleFilePicked(node, slot, file);
	};

	const btn = node.addWidget(
		"button",
		UPLOAD_NAME,
		"",
		() => {
			const slot = node.flags?.[TAB_FLAG] ?? "image_1";
			if (slot === ALL || !SLOTS.includes(slot)) return;
			input.accept = isVideoSlot(slot) ? VIDEO_ACCEPT : IMAGE_ACCEPT;
			input.click();
		},
		{ serialize: false, surfaces: { canvas: "shown", vueNode: "never", panel: "never" } },
	);
	btn.label = "choose file to upload";
}

/**
 * (Re)apply a tab: hide/show combos, upload button, preview, node size.
 */
function applyTab(node, slot) {
	if (!node.widgets?.length) return;
	if (!SLOTS.includes(slot) && slot !== ALL) slot = "image_1";
	node.flags = node.flags || {};
	node.flags[TAB_FLAG] = slot;

	for (const w of node.widgets) {
		if (SLOTS.includes(w.name)) {
			w.hidden = slot !== ALL && w.name !== slot;
		} else if (AUDIO_TOGGLES.includes(w.name)) {
			// each toggle rides along with its own video tab (or ALL)
			const videoSlot = w.name.replace(/_audio$/, "");
			w.hidden = slot !== ALL && videoSlot !== slot;
		}
	}

	// core upload widget (bound to image_1 by the frontend): only in ALL view
	const coreUpload = findWidget(node, "upload");
	if (coreUpload) coreUpload.hidden = slot !== ALL;

	// our per-slot upload button
	createUploadButton(node);
	const up = findWidget(node, UPLOAD_NAME);
	if (up) up.hidden = slot === ALL;

	if (slot === ALL) {
		removeWidget(node, PREVIEW_NAME);
		// let the default canvas preview system take over in ALL view
	} else {
		syncPreview(node, slot);
	}

	node.__pkMediaRefreshTabs?.();
	dirty(node);
	fitNode(node);
}

/**
 * Build the tab bar DOM widget (13 tabs: I1..I9, V1..V3, ALL).
 */
function buildTabBar(node) {
	if (findWidget(node, TAB_BAR_NAME)) return;

	const bar = document.createElement("div");
	bar.style.cssText =
		"display:flex;gap:2px;align-items:center;flex-wrap:nowrap;" +
		"height:22px;padding:0 0 4px 0;font-family:inherit;";

	const buttons = [];
	for (const slot of [...SLOTS, ALL]) {
		const b = document.createElement("button");
		b.textContent = slot === ALL ? "ALL" : comboLabel(slot);
		b.title = slot === ALL ? "Show all slots" : slot;
		b.style.cssText =
			"font-size:10px;line-height:1;padding:2px 5px;margin:0;" +
			"background:rgba(255,255,255,0.08);color:#bbb;border:none;" +
			"border-radius:3px;cursor:pointer;font-family:inherit;white-space:nowrap;";
		b.addEventListener("mouseenter", () => {
			if (b.dataset.active !== "1") b.style.background = "rgba(255,255,255,0.18)";
		});
		b.addEventListener("mouseleave", () => {
			if (b.dataset.active !== "1") b.style.background = "rgba(255,255,255,0.08)";
		});
		b.addEventListener("click", (ev) => {
			ev.stopPropagation?.();
			ev.preventDefault?.();
			applyTab(node, slot);
		});
		bar.appendChild(b);
		buttons.push({ slot, el: b });
	}

	const refreshActive = () => {
		const current = node.flags?.[TAB_FLAG] ?? "image_1";
		for (const { slot, el } of buttons) {
			const active = slot === current;
			el.dataset.active = active ? "1" : "0";
			el.style.background = active ? "#3b82f6" : "rgba(255,255,255,0.08)";
			el.style.color = active ? "#fff" : "#bbb";
		}
	};

	node.addDOMWidget(TAB_BAR_NAME, "string", bar, { serialize: false });
	// keep the tab bar pinned at the top of the widget list, at a FIXED
	// height — otherwise the layout treats it as a flexible widget and the
	// free space in a user-enlarged node pushes the combos down
	const w = findWidget(node, TAB_BAR_NAME);
	if (w) {
		node.widgets.splice(node.widgets.indexOf(w), 1);
		node.widgets.unshift(w);
		w.computeSize = () => [Math.max(200, (node.size?.[0] ?? 330) - 20), 26];
	}
	Object.defineProperty(node, "__pkMediaRefreshTabs", {
		value: refreshActive,
		writable: true,
		configurable: true,
		enumerable: false,
	});
	refreshActive();
}

/**
 * Guard: in tab mode, keep the preview matched to the active slot. The
 * frontend's media pipeline writes node.imgs asynchronously (and may stack
 * several entries → thumbnail strip), so re-assert ours periodically.
 * Idempotent and cheap.
 */
function guardPreview(node) {
	const slot = node.flags?.[TAB_FLAG];
	if (!slot || slot === ALL) return;
	syncPreview(node, slot);
	// video mode: keep the canvas preview widget away (it can't play mp4s)
	// and re-fit the player to the node's current (possibly pinned) size
	if (isVideoSlot(slot)) {
		if (findWidget(node, CANVAS_PREVIEW_NAME)) {
			removeWidget(node, CANVAS_PREVIEW_NAME);
			dirty(node);
		}
		findWidget(node, PREVIEW_NAME)?.__pkApplyDims?.();
	}
}

// Continuous guard: the media pipeline writes node.imgs asynchronously and
// (re-adds the canvas preview widget) at unpredictable times after load and
// after asset scans. In tab mode we own the preview area, so keep clearing
// the canvas preview artifacts. Cheap: a linear node scan every 300 ms.
setInterval(() => {
	if (typeof app === "undefined") return;
	const graph = app.canvas?.graph;
	if (!graph?._nodes) return;
	for (const node of graph._nodes) {
		if (node.type === "PK_MediaLoader") guardPreview(node);
	}
}, 300);

/**
 * Subgraph-aware downstream resolution — the same traversal the bundle
 * in/out nodes use (their resolveTarget): follow a link's target across
 * boundary nodes, hopping UP out of a SubgraphOutputNode (via the owning
 * SubgraphNode's parent-graph link) or DOWN into a SubgraphNode (via its
 * interior boundary input), until the real endpoint is reached.
 */
const MAX_BOUNDARY_HOPS = 32; // guard against corrupt/cyclic subgraph refs

function findNodeLoose(graph, id) {
	// graph.getNodeById() can miss nodes in subgraphs (type-sensitive Map
	// lookup) — fall back to a linear scan with loose id comparison
	if (id == null || !graph) return null;
	const direct = graph.getNodeById?.(id);
	if (direct) return direct;
	for (const node of graph._nodes || graph.nodes || []) {
		if (node.id === id || String(node.id) === String(id)) return node;
	}
	return null;
}

// Subgraphs carry no back-reference to the SubgraphNode that hosts them,
// so walk down from the root graph and match by object identity
function findOwningSubgraphNode(innerGraph) {
	const root = innerGraph?.rootGraph ?? app.graph;
	let found = null;
	(function walk(graph) {
		if (found || !graph) return;
		for (const node of graph._nodes || graph.nodes || []) {
			if (node.subgraph === innerGraph) {
				found = node;
				return;
			}
			if (node.subgraph) walk(node.subgraph);
			if (found) return;
		}
	})(root);
	return found;
}

function resolveBundlePeer(graph, id, slot, depth = 0) {
	if (id == null || !graph || depth > MAX_BOUNDARY_HOPS) return null;

	// outward: target is this graph's boundary output node -> hop up to the
	// owning SubgraphNode in the parent graph and continue on ITS link
	if (graph.outputNode && String(id) === String(graph.outputNode.id)) {
		const owner = findOwningSubgraphNode(graph);
		const ownerOutput = owner?.outputs?.[slot];
		const linkId = ownerOutput?.links?.[0];
		const link = linkId != null ? owner.graph.links[linkId] : null;
		if (!link) return null;
		return resolveBundlePeer(owner.graph, link.target_id, link.target_slot, depth + 1);
	}

	const node = findNodeLoose(graph, id);
	if (!node) return null;

	// inward: target is a SubgraphNode -> dive into its interior boundary
	// input node and continue on that link
	if (node.isSubgraphNode?.()) {
		const inner = node.subgraph;
		const inSlot = inner?.inputNode?.slots?.[slot];
		const linkId = inSlot?.linkIds?.[0];
		const link = linkId != null ? inner.links[linkId] : null;
		if (!link) return node; // fall back to the SubgraphNode's own port
		return resolveBundlePeer(inner, link.target_id, link.target_slot, depth + 1);
	}

	return node;
}

/**
 * Tell downstream bundle consumers how many slots this node provides, using
 * the same peer protocol as Bundle In/Out. A Bundle Out fed directly by the
 * loader has no Bundle In peer to size it, so it would stay collapsed at a
 * single output with nothing to connect the other 11 slots from. Resolution
 * crosses subgraph boundaries in both directions (loader inside / outside).
 */
function notifyBundlePeers(node) {
	const out = node.outputs?.find((o) => o.name === "bundle");
	if (!out?.links?.length || !node.graph?.links) return;
	for (const linkId of out.links) {
		const link = node.graph.links[linkId];
		if (!link) continue;
		const target = resolveBundlePeer(node.graph, link.target_id, link.target_slot);
		if (target?.receivePeerMinSlots) target.receivePeerMinSlots(BUNDLE_CHANNELS);
	}
}

app.registerExtension({
	name: "PlagueKind.PKMediaLoaderTabs",
	// new frontend: first arg is the node class, second the node def
	beforeRegisterNodeDef(nodeClass, nodeDef) {
		if (nodeDef?.name !== "PK_MediaLoader") return;

		const origOnNodeCreated = nodeClass.prototype.onNodeCreated;
		nodeClass.prototype.onNodeCreated = function () {
			origOnNodeCreated?.call(this);
			buildTabBar(this);
			applyTab(this, this.flags?.[TAB_FLAG] ?? "image_1");
			// slot combo changes refresh the preview (dropdown path; the
			// upload path also goes through combo.callback)
			for (const name of SLOTS) {
				const combo = findWidget(this, name);
				if (!combo) continue;
				const origCallback = combo.callback;
				combo.callback = (value) => {
					origCallback?.call(combo, value);
					if (this.flags?.[TAB_FLAG] === name) {
						syncPreview(this, name);
						dirty(this);
						fitNode(this);
					}
				};
			}
		};

		// live re-linking: tell Bundle Out peers the slot count as soon as
		// the bundle output gets wired (and keep it in sync on re-wiring)
		const origOnConnectionsChange = nodeClass.prototype.onConnectionsChange;
		nodeClass.prototype.onConnectionsChange = function () {
			origOnConnectionsChange?.apply(this, arguments);
			if (app.configuringGraph) return;
			notifyBundlePeers(this);
		};

		// Workflow restore: configure() runs after widgets + values exist.
		const origConfigure = nodeClass.prototype.configure;
		nodeClass.prototype.configure = function (data) {
			const r = origConfigure?.call(this, data);
			buildTabBar(this);
			applyTab(this, this.flags?.[TAB_FLAG] ?? "image_1");
			// the media pipeline writes node.imgs asynchronously after load —
			// re-assert the tab preview a few times to win the race
			const guard = () => {
				guardPreview(this);
				this.__pkMediaRefreshTabs?.();
			};
			setTimeout(guard, 150);
			setTimeout(guard, 600);
			setTimeout(guard, 2000);
			// links are restored after configure — notify bundle peers with a
			// few retries so the Bundle Out expands even mid-restore
			setTimeout(() => notifyBundlePeers(this), 300);
			setTimeout(() => notifyBundlePeers(this), 1500);
			setTimeout(() => notifyBundlePeers(this), 4000);
			return r;
		};
	},
});
