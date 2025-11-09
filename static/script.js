// Typewriter effect for terminal-style text
function typewriter(elementId, text, speed = 50) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.textContent = '';
    element.classList.add('terminal-cursor');
    
    let i = 0;
    const timer = setInterval(() => {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
        } else {
            clearInterval(timer);
            setTimeout(() => {
                element.classList.remove('terminal-cursor');
            }, 500);
        }
    }, speed);
}
