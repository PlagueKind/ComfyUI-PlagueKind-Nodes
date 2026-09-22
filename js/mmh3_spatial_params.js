// Front-end helper for MMH3SpatialSplitParams / LTX25SpatialSplitParams:
// auto-show/hide the two input groups depending on the tile_size_mode combo.
// Hidden widgets keep their stored values and serialize as usual - the server
// side already validates per mode, this is purely cosmetic.
import { app } from "../../scripts/app.js";

const TARGET_NODES = new Set(["MMH3SpatialSplitParams", "LTX25SpatialSplitParams"]);
const MODE_WIDGET = "tile_size_mode";
const SPECIFIC_WIDGETS = ["tile_width", "tile_height"];
const GRID_WIDGETS = ["upscale_width", "upscale_height", "grid_rows", "grid_cols"];
const VALID_MODES = new Set(["specific_size", "rows_cols", "auto"]);
const VALID_OVERLAP_MODES = new Set(["earlier", "later"]);
const VALID_BLEND_MODES = new Set(["linear", "smoothstep", "overwrite", "midpoint"]);

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
}

/**
 * Repair workflows saved before tile_size_mode/grid sizing was added.
 *
 * LiteGraph stores widget values positionally. When the seven grid widgets
 * were inserted ahead of the original nine settings, old values shifted into
 * the new fields: the old min_tile_size (normally 256) became grid_cols, and
 * overlap_mode/blend became the two overlap integers. ComfyUI then rejects the
 * prompt before the node can execute.
 *
 * Return true only for that exact legacy signature. Current workflows and
 * deliberately unusual values are left untouched.
 */
export function migrateLegacySpatialWidgets(node) {
    const required = [
        "upscale_width", "upscale_height", "tile_size_mode",
        "tile_width", "tile_height", "grid_rows", "grid_cols",
        "spatial_w_overlap", "spatial_h_overlap", "fade_width",
        "fade_height", "min_tile_size", "overlap_mode", "overlap_blend",
    ];
    const widgets = new Map((node.widgets || []).map((widget) => [widget.name, widget]));
    if (required.some((name) => !widgets.has(name))) return false;

    const mode = widgets.get("tile_size_mode").value;
    if (VALID_MODES.has(mode)) return false;

    const legacyValues = [
        widgets.get("upscale_width").value,
        widgets.get("upscale_height").value,
        mode,
        widgets.get("tile_width").value,
        widgets.get("tile_height").value,
        widgets.get("grid_rows").value,
        widgets.get("grid_cols").value,
    ];
    if (!legacyValues.every(finiteNumber)) return false;
    if (!VALID_OVERLAP_MODES.has(widgets.get("spatial_w_overlap").value)) return false;
    if (!VALID_BLEND_MODES.has(widgets.get("spatial_h_overlap").value)) return false;

    const legacy = {
        tile_width: widgets.get("upscale_width").value,
        tile_height: widgets.get("upscale_height").value,
        spatial_w_overlap: mode,
        spatial_h_overlap: widgets.get("tile_width").value,
        fade_width: widgets.get("tile_height").value,
        fade_height: widgets.get("grid_rows").value,
        min_tile_size: widgets.get("grid_cols").value,
        overlap_mode: widgets.get("spatial_w_overlap").value,
        overlap_blend: widgets.get("spatial_h_overlap").value,
    };

    widgets.get("upscale_width").value = 1024;
    widgets.get("upscale_height").value = 1024;
    widgets.get("tile_size_mode").value = "specific_size";
    widgets.get("tile_width").value = legacy.tile_width;
    widgets.get("tile_height").value = legacy.tile_height;
    widgets.get("grid_rows").value = 2;
    widgets.get("grid_cols").value = 2;
    widgets.get("spatial_w_overlap").value = legacy.spatial_w_overlap;
    widgets.get("spatial_h_overlap").value = legacy.spatial_h_overlap;
    widgets.get("fade_width").value = legacy.fade_width;
    widgets.get("fade_height").value = legacy.fade_height;
    widgets.get("min_tile_size").value = legacy.min_tile_size;
    widgets.get("overlap_mode").value = legacy.overlap_mode;
    widgets.get("overlap_blend").value = legacy.overlap_blend;
    return true;
}

function setWidgetVisible(node, name, visible) {
    const w = findWidget(node, name);
    if (!w) return;
    if ("hidden" in w) {
        // modern ComfyUI frontend: proper hidden flag, excluded from layout
        w.hidden = !visible;
        return;
    }
    // fallback for older frontends: park the widget as "converted"
    if (visible) {
        if (w.type === "converted-widget" && w.origType) {
            w.type = w.origType;
            if (w.origComputeOptions) w.computeOptions = w.origComputeOptions;
        }
    } else if (w.type !== "converted-widget") {
        w.origType = w.type;
        w.origComputeOptions = w.computeOptions;
        w.type = "converted-widget";
        w.computeOptions = () => {};
    }
}

function applyMode(node) {
    const mode = findWidget(node, MODE_WIDGET);
    if (!mode) return false;
    const specificSize = mode.value === "specific_size";
    const rowsCols = mode.value === "rows_cols";
    for (const name of SPECIFIC_WIDGETS) setWidgetVisible(node, name, specificSize);
    for (const name of GRID_WIDGETS) setWidgetVisible(node, name, rowsCols);
    if (node.onResize) node.onResize(node.size);
    app.graph.setDirtyCanvas(true, true);
    return true;
}

app.registerExtension({
    name: "MMH3UltimateUpscale.SpatialTileSizeMode",
    nodeCreated(node) {
        const cls = node.comfyClass || node.type;
        if (!TARGET_NODES.has(cls)) return;
        console.info("[MMH3-UltimateUpscale] tile_size_mode UI hook installed on", cls);

        const mode = findWidget(node, MODE_WIDGET);
        if (mode) {
            const origCallback = mode.callback;
            mode.callback = function () {
                const res = origCallback ? origCallback.apply(this, arguments) : undefined;
                applyMode(node);
                return res;
            };
        }
        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            const res = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            setTimeout(() => {
                if (migrateLegacySpatialWidgets(node)) {
                    console.warn("[MMH3-UltimateUpscale] migrated legacy spatial widget order on", cls);
                }
                applyMode(node);
            }, 0);
            return res;
        };
        setTimeout(() => {
            if (migrateLegacySpatialWidgets(node)) {
                console.warn("[MMH3-UltimateUpscale] migrated legacy spatial widget order on", cls);
            }
            applyMode(node);
        }, 0);
    },
});
