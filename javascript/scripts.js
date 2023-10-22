function getCurrentExtraSourceImg(dummy_component, imgCom) {
    const img = gradioApp().querySelector('#extras_image div div img');
    const removeButton = gradioApp().getElementById('nsfw_censor_mask').querySelector('button[aria-label="Remove Image"]');
    if (removeButton){
        removeButton.click();
    }
    return img ? [img.src] : [null];
}

function nudenetCensorApplyZoomAndPanIntegration () {
    if (typeof window.applyZoomAndPanIntegration === "function") {
        window.applyZoomAndPanIntegration("#nudenet_nsfw_censor_extras", ["#nsfw_censor_mask"]);
        var index = uiUpdateCallbacks.indexOf(nudenetCensorApplyZoomAndPanIntegration);
        if (index !== -1) {
            uiUpdateCallbacks.splice(index, 1);
        }
    }
}

onUiUpdate(nudenetCensorApplyZoomAndPanIntegration);
