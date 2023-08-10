const slider = document.querySelector(".image-comparison .slider");
const beforeImage = document.querySelector(".image-comparison .before-image");
const sliderLine = document.querySelector(".image-comparison .slider-line");
const sliderIcon = document.querySelector(".image-comparison .slider-icon");

const vid_slider = document.querySelector(".vid-comparison .vslider");
const beforeVid = document.querySelector(".vid-comparison .before-vid");
const vid_sliderLine = document.querySelector(".vid-comparison .vslider-line");
const vid_sliderIcon = document.querySelector(".vid-comparison .vslider-icon");

slider.addEventListener("input", (e) => {
    let sliderValue = e.target.value + "%";

    beforeImage.style.width = sliderValue;
    sliderLine.style.left = sliderValue;
    sliderIcon.style.left = sliderValue;
});

vid_slider.addEventListener("input", (e) => {
    let sliderValue = e.target.value + "%";

    beforeVid.style.width = sliderValue;
    vid_sliderLine.style.left = sliderValue;
    vid_sliderIcon.style.left = sliderValue;
});