import { app } from "../../scripts/app.js";

const MAX_SLOTS = 20;
const MAX_BOUNDARY_HOPS = 32; // guard against corrupt/cyclic subgraph refs

function slotLabel(prefix, i) {
  return `${prefix}_${i + 1}`;
}

// this.graph.getNodeById() can fail to resolve a node even when the link
// referencing it is present in this.graph.links -- observed in subgraphs,
// where node ids appear to be indexed (e.g. in a Map) in a way that a
// strict-equality / type-sensitive lookup misses. Fall back to a linear
// scan with loose id comparison. Note: this will legitimately return null
// for a boundary IO node id (SubgraphInputNode/SubgraphOutputNode) -- those
// aren't in the regular node list at all. Callers that need to cross a
// subgraph boundary should use resolveOrigin/resolveTarget below instead.
function findNodeById(graph, id) {
  if (id == null || !graph) return null;
  const direct = graph.getNodeById?.(id);
  if (direct) return direct;
  const list = graph._nodes || graph.nodes || [];
  for (const node of list) {
    if (node.id === id || String(node.id) === String(id)) return node;
  }
  return null;
}

// Subgraphs don't carry a back-reference to the SubgraphNode that hosts
// them, so to hop "up and out" of a boundary we walk down from the root
// graph and match by object identity -- the same pattern ComfyUI's own
// extension docs use for recursive traversal.
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

// Resolve a link's real upstream endpoint, crossing subgraph boundaries as
// needed. Two crossing cases, each of which can recurse arbitrarily deep:
//   - outward: the immediate origin is this graph's boundary input node
//     (SubgraphInputNode) -> hop to the owning SubgraphNode in the parent
//     graph and keep resolving from ITS input link at the same slot index.
//   - inward: the immediate origin is a SubgraphNode itself -> dive into
//     its interior graph's boundary output node and keep resolving there.
// crossInward controls whether the second case recurses or stops at the
// SubgraphNode's own port -- partner-node lookups need the real endpoint
// deep inside, but cosmetic labels should stay at the subgraph's exposed
// (friendly) name rather than surfacing internal implementation slots.
// Returns { node, graph, slot } of the resolved endpoint, or null.
function resolveOrigin(graph, id, slot, { depth = 0, crossInward = true } = {}) {
  if (id == null || !graph || depth > MAX_BOUNDARY_HOPS) return null;

  if (graph.inputNode && String(id) === String(graph.inputNode.id)) {
    const owner = findOwningSubgraphNode(graph);
    if (!owner || !owner.graph) return null; // exposed input isn't wired outside
    const ownerInput = owner.inputs?.[slot];
    if (!ownerInput || ownerInput.link == null) return null;
    const ownerLink = owner.graph.links[ownerInput.link];
    if (!ownerLink) return null;
    return resolveOrigin(owner.graph, ownerLink.origin_id, ownerLink.origin_slot, { depth: depth + 1, crossInward });
  }

  const node = findNodeById(graph, id);
  if (!node) return null;

  if (node.isSubgraphNode?.()) {
    if (!crossInward) return { node, graph, slot }; // stop at the exposed port
    const inner = node.subgraph;
    const outSlot = inner?.outputNode?.slots?.[slot];
    const innerLinkId = outSlot?.linkIds?.[0];
    const innerLink = innerLinkId != null ? inner.links[innerLinkId] : null;
    if (!innerLink) return { node, graph, slot }; // fall back to the SubgraphNode's own port
    return resolveOrigin(inner, innerLink.origin_id, innerLink.origin_slot, { depth: depth + 1, crossInward });
  }

  return { node, graph, slot };
}

// Symmetric to resolveOrigin: resolve a link's real downstream endpoint.
//   - outward: immediate target is this graph's boundary output node
//     (SubgraphOutputNode) -> hop to the owning SubgraphNode's output link.
//   - inward: immediate target is a SubgraphNode -> dive into its boundary
//     input node's link (only when crossInward is true).
function resolveTarget(graph, id, slot, { depth = 0, crossInward = true } = {}) {
  if (id == null || !graph || depth > MAX_BOUNDARY_HOPS) return null;

  if (graph.outputNode && String(id) === String(graph.outputNode.id)) {
    const owner = findOwningSubgraphNode(graph);
    if (!owner || !owner.graph) return null;
    const ownerOutput = owner.outputs?.[slot];
    const linkIds = ownerOutput?.links;
    if (!linkIds?.length) return null;
    // Bundle nodes only ever drive a single downstream link per slot.
    const ownerLink = owner.graph.links[linkIds[0]];
    if (!ownerLink) return null;
    return resolveTarget(owner.graph, ownerLink.target_id, ownerLink.target_slot, { depth: depth + 1, crossInward });
  }

  const node = findNodeById(graph, id);
  if (!node) return null;

  if (node.isSubgraphNode?.()) {
    if (!crossInward) return { node, graph, slot };
    const inner = node.subgraph;
    const inSlot = inner?.inputNode?.slots?.[slot];
    const innerLinkId = inSlot?.linkIds?.[0];
    const innerLink = innerLinkId != null ? inner.links[innerLinkId] : null;
    if (!innerLink) return { node, graph, slot };
    return resolveTarget(inner, innerLink.target_id, innerLink.target_slot, { depth: depth + 1, crossInward });
  }

  return { node, graph, slot };
}

// Labels stay at the nearest named boundary (a subgraph's own exposed
// port) rather than resolving all the way to whatever internal node
// happens to produce the value -- crossInward: false.
function describeOrigin(node, slotIndex) {
  const slot = node.inputs[slotIndex];
  const link = slot && node.graph.links[slot.link];
  if (!link) {
    console.warn("[BundleNodes] describeOrigin: no link on slot", { node: node.id, slotIndex, slotLink: slot?.link });
    return null;
  }
  const resolved = resolveOrigin(node.graph, link.origin_id, link.origin_slot, { crossInward: false });
  if (!resolved) {
    console.warn("[BundleNodes] describeOrigin: resolution failed", { node: node.id, slotIndex, origin_id: link.origin_id, origin_slot: link.origin_slot });
    return null;
  }
  const originSlot = resolved.node?.outputs?.[resolved.slot];
  if (!originSlot) {
    console.warn("[BundleNodes] describeOrigin: resolved node has no such output slot", { node: node.id, slotIndex, resolvedNode: resolved.node?.id, resolvedSlot: resolved.slot, outputsLen: resolved.node?.outputs?.length });
  }
  const rawName = originSlot ? (originSlot.name || "out") : null;
  // Strip dot-notation prefix and keep only the final segment
  return rawName ? rawName.split(".").pop() : null;
}

function describeTarget(node, slotIndex) {
  const slot = node.outputs[slotIndex];
  const linkIds = slot?.links;
  if (!linkIds || !linkIds.length) {
    console.warn("[BundleNodes] describeTarget: no links on slot", { node: node.id, slotIndex });
    return null;
  }
  const parts = linkIds
  .map((id) => {
    const link = node.graph.links[id];
    if (!link) {
      console.warn("[BundleNodes] describeTarget: link id not in graph.links", { node: node.id, slotIndex, linkId: id });
      return null;
    }
    const resolved = resolveTarget(node.graph, link.target_id, link.target_slot, { crossInward: false });
    if (!resolved) {
      console.warn("[BundleNodes] describeTarget: resolution failed", { node: node.id, slotIndex, target_id: link.target_id, target_slot: link.target_slot });
      return null;
    }
    const targetSlot = resolved.node?.inputs?.[resolved.slot];
    if (!targetSlot) {
      console.warn("[BundleNodes] describeTarget: resolved node has no such input slot", { node: node.id, slotIndex, resolvedNode: resolved.node?.id, resolvedSlot: resolved.slot, inputsLen: resolved.node?.inputs?.length });
    }
    const rawName = targetSlot ? (targetSlot.name || "in") : null;
    // Strip dot-notation prefix and keep only the final segment
    return rawName ? rawName.split(".").pop() : null;
  })
  .filter(Boolean);
  return parts.length ? parts.join(", ") : null;
}

app.registerExtension({
  name: "PlagueKind.BundleNodes",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "PlagueKindBundleIn") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        while (this.inputs.length > 1) this.removeInput(this.inputs.length - 1);
        this.inputs[0].name = slotLabel("input", 0);
        this.inputs[0].type = "*";
        this._peerMinSlots = 1;
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        onConfigure?.apply(this, arguments);
        setTimeout(() => {
          this._syncInputs();
          this._notifyDownstream();
        }, 0);
      };

      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function () {
        onConnectionsChange?.apply(this, arguments);
        // Prevent structural thrashing while ComfyUI is restoring the graph
        if (app.configuringGraph) return;
        this._syncInputs();
        this._notifyDownstream();
      };

      nodeType.prototype._getDownstreamOut = function () {
        const out = this.outputs?.[0];
        const linkId = out?.links?.[0];
        const link = linkId != null && this.graph.links[linkId];
        if (!link) return null;
        const resolved = resolveTarget(this.graph, link.target_id, link.target_slot);
        if (!resolved) {
          return null;
        }
        return resolved.node;
      };

      nodeType.prototype._syncInputs = function () {
        const partner = this._getDownstreamOut();
        for (let i = 0; i < this.inputs.length; i++) {
          const originDesc = describeOrigin(this, i);
          const targetDesc = partner ? describeTarget(partner, i) : null;

          // Prioritize destination name, fallback to source name
          const cleanName = targetDesc || originDesc;

          this.inputs[i].name = slotLabel("input", i);
          this.inputs[i].label = cleanName ? cleanName : slotLabel("input", i);
        }

        const last = this.inputs[this.inputs.length - 1];
        if (last.link != null && this.inputs.length < MAX_SLOTS) {
          this.addInput(slotLabel("input", this.inputs.length), "*");
        }
        const floor = Math.max(1, this._peerMinSlots || 1);
        while (
          this.inputs.length > floor &&
          this.inputs[this.inputs.length - 1].link == null &&
          this.inputs[this.inputs.length - 2].link == null
        ) {
          this.removeInput(this.inputs.length - 1);
        }
        this.setSize(this.computeSize());
        this.graph?.setDirtyCanvas?.(true, true);
        app.canvas?.setDirty?.(true, true);
      };

      nodeType.prototype.receivePeerMinSlots = function (n) {
        if (app.configuringGraph) return;
        this._peerMinSlots = Math.min(MAX_SLOTS, Math.max(1, n));
        while (this.inputs.length < this._peerMinSlots) {
          this.addInput(slotLabel("input", this.inputs.length), "*");
        }
        this._syncInputs();
      };

      nodeType.prototype._notifyDownstream = function () {
        const target = this._getDownstreamOut();
        target?.receivePeerMinSlots?.(this.inputs.length);
        target?._syncOutputs?.();
      };
    }

    if (nodeData.name === "PlagueKindBundleOut") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        while (this.outputs.length > 1) this.removeOutput(this.outputs.length - 1);
        this.outputs[0].name = slotLabel("output", 0);
        this.outputs[0].type = "*";
        this._peerMinSlots = 1;
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        onConfigure?.apply(this, arguments);
        setTimeout(() => {
          this._syncOutputs();
          this._notifyUpstream();
        }, 0);
      };

      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function () {
        onConnectionsChange?.apply(this, arguments);
        if (app.configuringGraph) return;
        this._syncOutputs();
        this._notifyUpstream();
      };

      nodeType.prototype._getUpstreamIn = function () {
        const slot = this.inputs[0];
        const link = slot && this.graph.links[slot.link];
        if (!link) return null;
        const resolved = resolveOrigin(this.graph, link.origin_id, link.origin_slot);
        if (!resolved) {
          return null;
        }
        return resolved.node;
      };

      nodeType.prototype._syncOutputs = function () {
        const partner = this._getUpstreamIn();
        for (let i = 0; i < this.outputs.length; i++) {
          const targetDesc = describeTarget(this, i);
          const originDesc = partner ? describeOrigin(partner, i) : null;

          // FIX: Prioritize this node's own downstream target name.
          // Only fall back to the upstream source name if this output isn't plugged into anything yet.
          const cleanName = targetDesc || originDesc;

          this.outputs[i].name = slotLabel("output", i);
          this.outputs[i].label = cleanName ? cleanName : slotLabel("output", i);
        }

        const last = this.outputs[this.outputs.length - 1];
        if (last.links?.length && this.outputs.length < MAX_SLOTS) {
          this.addOutput(slotLabel("output", this.outputs.length), "*");
        }
        const floor = Math.max(1, this._peerMinSlots || 1);
        while (
          this.outputs.length > floor &&
          !this.outputs[this.outputs.length - 1].links?.length &&
          !this.outputs[this.outputs.length - 2].links?.length
        ) {
          this.removeOutput(this.outputs.length - 1);
        }
        this.setSize(this.computeSize());
        this.graph?.setDirtyCanvas?.(true, true);
        app.canvas?.setDirty?.(true, true);
      };

      nodeType.prototype.receivePeerMinSlots = function (n) {
        if (app.configuringGraph) return;
        this._peerMinSlots = Math.min(MAX_SLOTS, Math.max(1, n));
        while (this.outputs.length < this._peerMinSlots) {
          this.addOutput(slotLabel("output", this.outputs.length), "*");
        }
        this._syncOutputs();
      };

      nodeType.prototype._notifyUpstream = function () {
        const upstream = this._getUpstreamIn();
        upstream?.receivePeerMinSlots?.(this.outputs.length);
        upstream?._syncInputs?.();
      };
    }
  },
});
