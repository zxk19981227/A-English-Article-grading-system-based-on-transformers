import "leaflet/dist/leaflet.css"
import $L from "leaflet";

const createMap = (divId, options) => {
    let map = $L.map(divId, options);
    return map;
};

export default { createMap };
