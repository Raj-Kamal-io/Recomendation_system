document.addEventListener('DOMContentLoaded', function () {
    // Dark/Light Mode Toggler
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) {
        // Load initial theme state from local storage
        const activeTheme = localStorage.getItem('theme');
        if (activeTheme === 'dark') {
            document.body.classList.add('dark-mode');
            themeToggleBtn.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            document.body.classList.remove('dark-mode');
            themeToggleBtn.innerHTML = '<i class="fas fa-moon"></i>';
        }

        themeToggleBtn.addEventListener('click', function () {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
            themeToggleBtn.innerHTML = isDark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
        });
    }
});
