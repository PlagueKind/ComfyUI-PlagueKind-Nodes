import { app } from "../../scripts/app.js";

const NODE_TYPE = "H3SLAAttention";

app.registerExtension({
    name: "PlagueKind.H3SLAAttention.ReferenceProtection",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            const node = this;

            function applyReferenceLabel() {
                const referenceMode = node.widgets?.find(
                    widget => widget.name === "reference_protection"
                );
                const protectAudio = node.widgets?.find(
                    widget => widget.name === "protect_audio"
                );
                if (referenceMode) {
                    referenceMode.label = "Protect Image/Video Reference";
                    if (String(referenceMode.value).toLowerCase() === "manual") {
                        referenceMode.value = "Light";
                    }
                }
                if (protectAudio && typeof protectAudio.value === "string") {
                    const disabled = ["off", "false", "0", "no"].includes(
                        protectAudio.value.toLowerCase()
                    );
                    protectAudio.value = !disabled;
                }
                node.graph?.setDirtyCanvas(true, true);
            }

            requestAnimationFrame(applyReferenceLabel);
            return result;
        };
    },
});
