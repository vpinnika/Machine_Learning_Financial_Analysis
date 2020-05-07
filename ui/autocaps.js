
var inputs = document.querySelectorAll('input');
inputs.forEach((input)=>{
    input.addEventListener('input',(e)=>{
        e.target.value = e.target.value.toUpperCase();
    })
});