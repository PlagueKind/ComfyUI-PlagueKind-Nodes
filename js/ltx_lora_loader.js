/**
 * LTX / MiniMax H3 LoRA Loader - Dynamic Height Edition (Android & Mobile Layout Hardened)
 */
import { app } from "../../scripts/app.js";
const NODE_TYPE = "LTX_lora_loader";
const MAX_SLOTS = 10;
const MIN_SIZE = [420, 240];
let _loraCache = ["None"];
// Every on-canvas node instance registers a refresh callback here so its
// dropdown/warning state can be updated live, without needing a new node
// to be created or the page to be reloaded.
const _liveInstances = new Set();

function _notifyLiveInstances() {
    for (const refresh of _liveInstances) {
        try { refresh(); } catch (e) { console.warn("LoRA instance refresh failed", e); }
    }
}

async function getLoraList(nodeData) {
    try {
        const list = nodeData?.input?.hidden?.available_loras?.[0] || nodeData?.input?.required?.lora_name?.[0];
        if (Array.isArray(list)) {
            _loraCache = ["None", ...list];
            _notifyLiveInstances();
        }
    } catch (e) { console.warn("LoRA fetch failed", e); }
}

function loraBasename(fullPath) {
    if (!fullPath || fullPath === "None") return "None";
    const base = fullPath.split(/[/\\]/).pop();
    return base.replace(/\.[^.]+$/, "");
}

function loraFolder(fullPath) {
    if (!fullPath || fullPath === "None") return "";
    const parts = fullPath.split(/[/\\]/);
    return parts.length > 1 ? parts[parts.length - 2] : "";
}

function loraDisplayName(fullPath, allSlots) {
    const name = loraBasename(fullPath);
    if (!allSlots || fullPath === "None") return name;
    const siblings = allSlots.filter(s => {
        const lora = s.getLora?.();
        return lora && lora !== fullPath && loraBasename(lora) === name;
    });
    if (siblings.length > 0) {
        const folder = loraFolder(fullPath);
        return folder ? `${name} (${folder})` : name;
    }
    return name;
}

app.registerExtension({
    name: "PlagueKind.LTX_lora_loader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        // Runs every time node defs are (re)registered, including on
        // "Refresh node definitions" -- not just when a new node instance
        // is created -- so the dropdown cache actually picks up new loras.
        await getLoraList(nodeData);
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            orig?.apply(this, arguments);
            const node = this;
            const ROW_H = 40;
            const BTN_H = 60;
            function minHeight() {
                return Math.max(MIN_SIZE[1], (slots.length * ROW_H) + BTN_H);
            }
            node.onResize = function (size) {
                size[0] = Math.max(MIN_SIZE[0], size[0]);
                size[1] = Math.max(minHeight(), size[1]);
                this.size = size;
            };
            node.properties = node.properties || {};
            const stackIndex = node.widgets.findIndex(w => w.name === "stack_data");
            let stackWidget = stackIndex !== -1 ? node.widgets[stackIndex] : null;
            if (stackWidget) {
                stackWidget.computeSize = () => [0, -4];
                stackWidget.draw = () => {};
            }
            const modeWidget = node.widgets.find(w => w.name === "mode");
            function applyModeVisibility() {
                const m = modeWidget ? modeWidget.value : "normal";
                // MiniMax H3 loras seen so far carry no video_patch_proj/audio_patch_proj
                // keys, so V/A are dead weight in that mode - keep them LTX-only for now.
                const showVA = m === "ltx";
                const showT = m === "minimax";
                for (const s of slots) s.setVisibility?.(showVA, showT);
                syncSize();
            }
            if (modeWidget) {
                const origModeCallback = modeWidget.callback;
                modeWidget.callback = function (...args) {
                    const r = origModeCallback?.apply(this, args);
                    applyModeVisibility();
                    return r;
                };
            }
            let initialData = [];
            try {
                const raw = node.properties["stack_data"] || stackWidget?.value || "[]";
                initialData = JSON.parse(raw);
            } catch {}
            const container = document.createElement("div");
            Object.assign(container.style, {
                display: "flex", flexDirection: "column", gap: "4px",
                width: "100%", padding: "4px", boxSizing: "border-box",
                fontFamily: "var(--font)", color: "var(--fg-color)"
            });
            const inputStyle = `
            background: var(--comfy-input-bg);
            color: var(--input-text);
            border: 1px solid var(--border-color);
            border-radius: 4px; padding: 2px 4px;
            font-size: 10px; outline: none;
            transition: all 0.1s ease;
            box-sizing: border-box;
            `;
            let slots = [];
            let _rafPending = false;
            function syncSize() {
                if (_rafPending) return;
                _rafPending = true;
                requestAnimationFrame(() => {
                    _rafPending = false;
                    const targetH = Math.max(minHeight(), container.scrollHeight + 20);
                    node.setSize([Math.max(node.size[0], MIN_SIZE[0]), targetH]);
                });
            }
            function refreshAllDisplayNames() {
                for (const s of slots) s.refreshDisplayName?.();
            }
            function refreshLoraState() {
                for (const s of slots) {
                    s.checkMissing?.();
                    s.refreshDisplayName?.();
                }
            }
            _liveInstances.add(refreshLoraState);
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                _liveInstances.delete(refreshLoraState);
                origOnRemoved?.apply(this, arguments);
            };
            function syncData() {
                const data = slots.map(s => s.getValue());
                const json = JSON.stringify(data);
                if (stackWidget) stackWidget.value = json;
                node.properties["stack_data"] = json;
                node.onPropertyChanged?.("stack_data", json);
                refreshAllDisplayNames();
                syncSize();
            }
            async function refreshCache() {
                try {
                    const res = await fetch("/plaguekind/ltx_lora_loader/refresh", {
                        method: "GET",
                        headers: { "Content-Type": "application/json" }
                    });
                    const data = await res.json();
                    if (data.loras) {
                        _loraCache = ["None", ...data.loras];
                        slots.forEach(s => s.checkMissing?.());
                    }
                } catch (e) {
                    console.error(e);
                }
            }
            function sortTree(items) {
                items.sort((a, b) => {
                    if (a.content === "None") return -1;
                    if (b.content === "None") return 1;
                    if (a.has_submenu && !b.has_submenu) return -1;
                    if (!a.has_submenu && b.has_submenu) return 1;
                    return a.content.localeCompare(b.content);
                });
                for (const item of items) {
                    if (item.has_submenu && item.submenu?.options) sortTree(item.submenu.options);
                }
            }
            function buildMenuTree(list, onSelect) {
                const root = [];
                let noneAdded = false;
                for (const item of list) {
                    if (item === "None") {
                        if (!noneAdded) {
                            root.push({ content: "None", callback: () => onSelect("None") });
                            noneAdded = true;
                        }
                        continue;
                    }
                    const parts = item.split(/[/\\]/);
                    let current = root;
                    for (let i = 0; i < parts.length; i++) {
                        const part = parts[i];
                        let existing = current.find(x => x.content === "📁 " + part || x.content === part);
                        if (!existing && i !== parts.length - 1) {
                            existing = { content: "📁 " + part, has_submenu: true, submenu: { options: [] } };
                            current.push(existing);
                        } else if (i === parts.length - 1) {
                            const fullPath = item;
                            current.push({ content: part, callback: () => onSelect(fullPath) });
                        }
                        current = existing?.submenu?.options || current;
                    }
                }
                sortTree(root);
                return root;
            }
            function openLoraMenu(e, onSelect) {
                const list = _loraCache;
                const searchIndex = [];
                let noneAdded = false;
                for (const item of list) {
                    if (item === "None") {
                        if (!noneAdded) {
                            searchIndex.push({ display: "None", fullPath: "None", isFolder: false });
                            noneAdded = true;
                        }
                        continue;
                    }
                    const parts = item.split(/[/\\]/);
                    for (let i = 1; i < parts.length; i++) {
                        const folderPath = parts.slice(0, i).join("/");
                        if (!searchIndex.find(x => x.display === "📁 " + folderPath)) {
                            searchIndex.push({ display: "📁 " + folderPath, fullPath: null, isFolder: true, folderPrefix: folderPath });
                        }
                    }
                    searchIndex.push({ display: item.split(/[/\\]/).pop(), fullPath: item, isFolder: false });
                }
                const tree = buildMenuTree(list, onSelect);
                const menu = new LiteGraph.ContextMenu(tree, { event: e, scale: 1.2 });
                requestAnimationFrame(() => {
                    const root = menu?.root;
                    if (!root) return;
                    const header = document.createElement("div");
                    header.style.cssText = "display:flex;justify-content:space-between;align-items:center;padding:5px 8px;border-bottom:1px solid #444;";
                    const box = document.createElement("input");
                    box.placeholder = "Search LoRA...";
                    box.style.cssText = `flex:1;padding:4px;background:#222;color:white;border:1px solid #444;border-radius:4px;font-size:12px;`;
                    const refreshBtn = document.createElement("button");
                    refreshBtn.innerHTML = "🔄";
                    refreshBtn.title = "Refresh LoRA Cache";
                    refreshBtn.style.cssText = "margin-left:6px;padding:2px 6px;background:#333;border:none;border-radius:3px;cursor:pointer;";
                    refreshBtn.onclick = (ev) => {
                        ev.stopPropagation();
                        refreshCache();
                        menu.close?.();
                    };
                    header.append(box, refreshBtn);
                    root.prepend(header);
                    const flatList = document.createElement("div");
                    flatList.style.cssText = "display:none;max-height:320px;overflow-y:auto;min-width:320px;width:100%;box-sizing:border-box;";
                    root.appendChild(flatList);
                    const treeEntries = Array.from(root.querySelectorAll(".litemenu-entry"));
                    let currentFolderFilter = null;
                    function renderSearchResults(q) {
                        flatList.innerHTML = "";
                        const matches = searchIndex.filter(entry => {
                            if (currentFolderFilter && !entry.isFolder) {
                                return entry.fullPath && entry.fullPath.startsWith(currentFolderFilter + "/");
                            }
                            return entry.display.toLowerCase().includes(q);
                        });
                        for (const entry of matches) {
                            const el = document.createElement("div");
                            el.className = "litemenu-entry";
                            el.textContent = entry.display;
                            el.style.cssText = "padding:4px 8px;cursor:pointer;font-size:12px;white-space:nowrap;width:100%;box-sizing:border-box;";
                            if (entry.isFolder) {
                                el.style.fontStyle = "italic";
                                el.addEventListener("mousedown", (ev) => {
                                    ev.preventDefault();
                                    ev.stopPropagation();
                                    currentFolderFilter = entry.folderPrefix;
                                    box.value = entry.folderPrefix + "/";
                                    renderSearchResults(box.value.toLowerCase().trim());
                                });
                            } else {
                                el.addEventListener("mousedown", (ev) => {
                                    ev.preventDefault();
                                    ev.stopPropagation();
                                    onSelect(entry.fullPath);
                                    menu.close?.();
                                });
                            }
                            flatList.appendChild(el);
                        }
                    }
                    box.addEventListener("input", () => {
                        const q = box.value.toLowerCase().trim();
                        treeEntries.forEach(el => el.style.display = q ? "none" : "");
                        flatList.style.display = q ? "block" : "none";
                        renderSearchResults(q);
                    });
                    box.focus();
                });
            }
            function makeDraggable(row, slotObj) {
                row.draggable = true;
                row.addEventListener("dragstart", (e) => {
                    e.dataTransfer.setData("text/plain", slots.indexOf(slotObj));
                    row.style.opacity = "0.3";
                });
                row.addEventListener("dragend", () => {
                    slotObj.updateRowState?.();
                });
                row.addEventListener("dragover", (e) => e.preventDefault());
                row.addEventListener("drop", (e) => {
                    e.preventDefault();
                    const fromIndex = parseInt(e.dataTransfer.getData("text/plain"));
                    const toIndex = slots.indexOf(slotObj);
                    if (fromIndex === toIndex) return;
                    const [moved] = slots.splice(fromIndex, 1);
                    slots.splice(toIndex, 0, moved);
                    slots.forEach(s => s.row.remove());
                    slots.forEach(s => container.appendChild(s.row));
                    syncData();
                });
            }
            function addSlot(data = { on: true, lora: "None", str: 1.0, v: 1.0, a: 1.0, t: 1.0 }) {
                if (slots.length >= MAX_SLOTS) return;
                // backfill fields missing from rows saved before this field existed,
                // so a stale/partial row doesn't silently default a slider to 0
                data = { on: true, lora: "None", str: 1.0, v: 1.0, a: 1.0, t: 1.0, ...data };
                const row = document.createElement("div");
                row.style.cssText = "display:flex;align-items:center;gap:6px;width:100%;min-height:28px;background:var(--comfy-menu-bg);padding:4px;border-radius:4px;border:1px solid var(--border-color);transition:all 0.15s ease;box-sizing:border-box;";

                const handle = document.createElement("div");
                handle.textContent = "⋮";
                handle.style.cssText = "color:#777;font-size:18px;cursor:grab;padding:0 0px;user-select:none;width:10px;text-align:center;flex-shrink:0;";

                handle.addEventListener("mouseenter", () => {
                    if (!chk.checked) return;
                    row.style.background = "var(--comfy-input-bg)";
                    row.style.borderColor = "var(--primary-color)";
                });
                handle.addEventListener("mouseleave", () => {
                    if (!chk.checked) return;
                    row.style.background = "var(--comfy-menu-bg)";
                    row.style.borderColor = "var(--border-color)";
                });

                const chk = document.createElement("input");
                chk.type = "checkbox"; chk.checked = data.on;
                chk.style.flexShrink = "0";

                function updateRowState() {
                    const targets = [handle, sel, str.wrap, v.wrap, a.wrap, t.wrap, rm];
                    if (chk.checked) {
                        row.style.opacity = "1";
                        row.style.filter = "none";
                        targets.forEach(el => el.style.pointerEvents = "auto");
                    } else {
                        row.style.opacity = "0.45";
                        row.style.filter = "grayscale(75%)";
                        targets.forEach(el => el.style.pointerEvents = "none");
                    }
                }

                chk.onchange = () => {
                    updateRowState();
                    syncData();
                };

                const sel = document.createElement("div");
                sel.setAttribute("role", "button");
                sel.dataset.lora = data.lora;
                sel.style.cssText = inputStyle + "flex-grow:1;min-width:0;width:0;flex-shrink:1;display:flex;align-items:center;justify-content:space-between;cursor:pointer;overflow:hidden;user-select:none;white-space:nowrap;";

                sel.addEventListener("mouseenter", () => {
                    if (!chk.checked) return;
                    sel.style.background = "var(--comfy-menu-bg)";
                    sel.style.borderColor = "var(--primary-color)";
                });
                sel.addEventListener("mouseleave", () => {
                    if (!chk.checked) return;
                    sel.style.background = "var(--comfy-input-bg)";
                    checkMissing();
                });

                const selText = document.createElement("span");
                selText.textContent = loraDisplayName(data.lora, slots);
                selText.title = data.lora;
                selText.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex-grow:1;text-align:left;min-width:0;margin-right:4px;";

                const warn = document.createElement("span");
                warn.textContent = "⚠";
                warn.style.cssText = "color:var(--error-text, #ff5555);font-size:11px;margin-right:2px;flex-shrink:0;display:none;font-weight:bold;";

                function checkMissing() {
                    const fullPath = sel.dataset.lora;
                    if (!fullPath || fullPath === "None") {
                        warn.style.display = "none";
                        sel.style.borderColor = "var(--border-color)";
                        return;
                    }

                    const norm = p => p ? p.replace(/[/\\]/g, "/").toLowerCase() : "";
                    const isMissing = !_loraCache.some(x => norm(x) === norm(fullPath));

                    if (isMissing) {
                        warn.style.display = "inline";
                        warn.title = `File missing from environment:\n${fullPath}`;
                        sel.style.borderColor = "var(--error-text, #ff5555)";
                    } else {
                        warn.style.display = "none";
                        sel.style.borderColor = "var(--border-color)";
                    }
                }

                const arrow = document.createElement("span");
                arrow.textContent = "▼";
                arrow.style.cssText = "flex-shrink:0;font-size:8px;opacity:0.6;";

                sel.append(warn, selText, arrow);
                sel.onclick = (e) => {
                    if (!chk.checked) return;
                    openLoraMenu(e, (fullPath) => {
                        sel.dataset.lora = fullPath;
                        selText.textContent = loraDisplayName(fullPath, slots);
                        selText.title = fullPath;
                        checkMissing();
                        syncData();
                    });
                };
                function showNumPopup(currentVal, label, onConfirm) {
                    const overlay = document.createElement("div");
                    overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:9999;display:flex;align-items:center;justify-content:center;";
                    const panel = document.createElement("div");
                    panel.style.cssText = "background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:8px;padding:16px;display:flex;flex-direction:column;gap:10px;min-width:200px;box-shadow:0 4px 24px rgba(0,0,0,0.5);";
                    const title = document.createElement("div");
                    title.textContent = `Set ${label.replace(":", "")}`;
                    title.style.cssText = "font-size:13px;font-weight:bold;color:var(--fg-color);";

                    const popInp = document.createElement("input");
                    popInp.type = "text"; popInp.inputMode = "decimal"; popInp.value = currentVal;
                    popInp.style.cssText = inputStyle + "width:100%;box-sizing:border-box;font-size:14px;padding:6px 8px;";

                    const btnRow = document.createElement("div");
                    btnRow.style.cssText = "display:flex;gap:8px;justify-content:flex-end;";
                    const cancel = document.createElement("button");
                    cancel.textContent = "Cancel";
                    cancel.style.cssText = inputStyle + "cursor:pointer;padding:4px 12px;";
                    const ok = document.createElement("button");
                    ok.textContent = "OK";
                    ok.style.cssText = inputStyle + "cursor:pointer;padding:4px 12px;font-weight:bold;";
                    const close = () => document.body.removeChild(overlay);
                    cancel.onclick = close;
                    ok.onclick = () => {
                        const v = parseFloat(popInp.value);
                        if (!isNaN(v)) onConfirm(v);
                        close();
                    };
                    popInp.addEventListener("keydown", (e) => {
                        if (e.key === "Enter") ok.click();
                        if (e.key === "Escape") close();
                    });
                        overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
                        btnRow.append(cancel, ok);
                        panel.append(title, popInp, btnRow);
                        overlay.appendChild(panel);
                        document.body.appendChild(overlay);
                        requestAnimationFrame(() => { popInp.focus(); popInp.select(); });
                }
                function makeEffectiveTooltip(strInp, multInp, multLabel) {
                    const s = parseFloat(strInp.value) || 0;
                    const m = parseFloat(multInp.value) || 0;
                    return `${multLabel.replace(":", "")} ${s.toFixed(2)} × ${m.toFixed(2)} = ${(s * m).toFixed(3)} (effective)`;
                }
                function num(val, label) {
                    const wrap = document.createElement("div");
                    wrap.style.cssText = "display:flex;align-items:center;gap:2px;flex-shrink:0;";
                    const lbl = document.createElement("span");
                    lbl.textContent = label;
                    lbl.style.fontSize = "10px";

                    const inp = document.createElement("input");
                    inp.type = "text"; inp.inputMode = "decimal"; inp.value = Number.isFinite(Number(val)) ? Number(val).toFixed(2) : "0.00";
                    inp.style.cssText = inputStyle + "width:33px;text-align:center;flex-shrink:0;box-sizing:border-box;";

                    inp.addEventListener("change", syncData);
                    inp.addEventListener("input", syncData);
                    inp.addEventListener("click", (e) => {
                        if (!chk.checked) return;
                        e.preventDefault();
                        e.stopPropagation();
                        showNumPopup(inp.value, label, (newVal) => {
                            inp.value = newVal.toFixed(2);
                            syncData();
                        });
                    });
                    wrap.append(lbl, inp);
                    return { wrap, inp };
                }
                const str = num(data.str, "S:");
                const v = num(data.v, "V:");
                const a = num(data.a, "A:");
                const t = num(data.t, "T:");
                function setVisibility(showVA, showT) {
                    v.wrap.style.display = showVA ? "" : "none";
                    a.wrap.style.display = showVA ? "" : "none";
                    t.wrap.style.display = showT ? "" : "none";
                }
                function updateTooltips() {
                    v.inp.title = makeEffectiveTooltip(str.inp, v.inp, "V") + " (no-op unless the LoRA has video-specific keys)";
                    a.inp.title = makeEffectiveTooltip(str.inp, a.inp, "A") + " (no-op unless the LoRA has audio-specific keys)";
                    t.inp.title = makeEffectiveTooltip(str.inp, t.inp, "T") + " (MiniMax H3 text I/O only - no effect on LTX loras)";
                    str.inp.title = `S: ${(parseFloat(str.inp.value) || 0).toFixed(2)} (master strength)`;
                }
                updateTooltips();
                str.inp.addEventListener("input", updateTooltips);
                v.inp.addEventListener("input", updateTooltips);
                a.inp.addEventListener("input", updateTooltips);
                t.inp.addEventListener("input", updateTooltips);

                const rm = document.createElement("button");
                rm.innerHTML = "✖";
                rm.style.cssText = "background:transparent;color:var(--error-text);border:1px solid transparent;cursor:pointer;font-size:12px;padding:2px 6px;border-radius:4px;transition:all 0.1s ease;flex-shrink:0;";

                rm.addEventListener("mouseenter", () => {
                    rm.style.background = "var(--comfy-menu-bg)";
                    rm.style.borderColor = "var(--primary-color)";
                    rm.style.color = "#ff5555";
                });
                rm.addEventListener("mouseleave", () => {
                    rm.style.background = "transparent";
                    rm.style.borderColor = "transparent";
                    rm.style.color = "var(--error-text)";
                });

                const slotObj = {
                    row: row,
                    getValue: () => ({
                        on: chk.checked,
                        lora: sel.dataset.lora,
                        str: parseFloat(str.inp.value) || 0.0,
                                     v: parseFloat(v.inp.value) || 0.0,
                                     a: parseFloat(a.inp.value) || 0.0,
                                     t: parseFloat(t.inp.value) || 0.0
                    }),
                    setVisibility: setVisibility,
                    getLora: () => sel.dataset.lora,
                      refreshDisplayName: () => {
                          const lora = sel.dataset.lora;
                          selText.textContent = loraDisplayName(lora, slots);
                      },
                      checkMissing: checkMissing,
                      updateRowState: updateRowState,
                      remove: () => { row.remove(); slots = slots.filter(s => s !== slotObj); syncData(); }
                };
                rm.onclick = slotObj.remove;
                row.append(handle, chk, sel, str.wrap, v.wrap, a.wrap, t.wrap, rm);
                slots.push(slotObj);
                makeDraggable(row, slotObj);
                container.appendChild(row);

                checkMissing();
                updateRowState();
                applyModeVisibility();
                syncData();
            }
            const addBtn = document.createElement("button");
            addBtn.textContent = "＋ Add LoRA";
            addBtn.style.cssText = inputStyle + "width:100%;cursor:pointer;font-weight:bold;transition:all 0.1s ease;";

            addBtn.addEventListener("mouseenter", () => {
                addBtn.style.background = "var(--comfy-menu-bg)";
                addBtn.style.borderColor = "var(--primary-color)";
            });
            addBtn.addEventListener("mouseleave", () => {
                addBtn.style.background = "var(--comfy-input-bg)";
                addBtn.style.borderColor = "var(--border-color)";
            });

            addBtn.onclick = () => addSlot();
            container.appendChild(addBtn);
            node.addDOMWidget("lora_ui", "HTML", container);
            initialData.forEach(d => addSlot(d));
            requestAnimationFrame(syncSize);
            const _origConfigure = node.configure;
            node.configure = function (data) {
                if (_origConfigure) _origConfigure.call(node, data);
                slots.forEach(s => s.remove());
                slots = [];
                try {
                    const raw = data?.properties?.["stack_data"] || "[]";
                    if (stackWidget) stackWidget.value = raw;
                    JSON.parse(raw).forEach(d => addSlot(d));
                } catch {}
                requestAnimationFrame(syncSize);
            };
        };
    }
});
