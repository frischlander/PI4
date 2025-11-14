// Main JavaScript for Dengue Analysis Project

document.addEventListener('DOMContentLoaded', function() {
    // Load summary data
    loadSummaryData();
    
    // Initialize prediction form
    initializePredictionForm();
    
    // Smooth scrolling for navigation links
    initializeSmoothScrolling();
});

// Load summary data from API
async function loadSummaryData() {
    try {
        const response = await fetch('/api/data/summary');
        const data = await response.json();
        
        // Update summary cards
        document.getElementById('total-casos').textContent = data.total_casos.toLocaleString('pt-BR');
        document.getElementById('casos-hospitalizados').textContent = data.casos_hospitalizados.toLocaleString('pt-BR');
        document.getElementById('anos-cobertura').textContent = data.anos_cobertura; // Adicionado
        
        // Update other elements if they exist
        const totalCasosElement = document.getElementById('total-casos-dash');
        if (totalCasosElement) {
            totalCasosElement.textContent = data.total_casos.toLocaleString('pt-BR');
        }
        
    } catch (error) {
        console.error('Erro ao carregar dados de resumo:', error);
    }
}

// Initialize prediction form
function initializePredictionForm() {
    const form = document.getElementById('prediction-form');
    if (!form) return;
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const button = form.querySelector('button[type="submit"]');
        const originalText = button.innerHTML;
        
        // Show loading state
        button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Calculando...';
        button.disabled = true;
        
        try {
            // Collect form data
            const formData = {
                IDADE: parseInt(document.getElementById('idade').value),
                CS_SEXO: document.getElementById('sexo').value,
                FEBRE: document.getElementById('febre').value,
                VOMITO: document.getElementById('vomito').value,
                MIALGIA: document.getElementById('mialgia').value,
                CEFALEIA: document.getElementById('cefaleia').value,
                EXANTEMA: document.getElementById('exantema').value
                // Outras features categóricas serão preenchidas com valores padrão no app.py
            };
            
            // Make prediction request
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                // Show result
                const probabilityValue = (result.probabilidade_hospitalizacao * 100).toFixed(2);
                document.getElementById('probability-value').textContent = probabilityValue + '%';
                document.getElementById('prediction-result').style.display = 'block';
                
                // Scroll to result
                document.getElementById('prediction-result').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'nearest' 
                });
            } else {
                throw new Error(result.error || 'Erro na predição');
            }
            
        } catch (error) {
            console.error('Erro na predição:', error);
            alert('Erro ao calcular a probabilidade. Tente novamente: ' + error.message);
        } finally {
            // Restore button state
            button.innerHTML = originalText;
            button.disabled = false;
        }
    });
}

// Initialize smooth scrolling
function initializeSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Utility function to format numbers
function formatNumber(num) {
    return num.toLocaleString('pt-BR');
}

// Utility function to format percentage
function formatPercentage(num, decimals = 2) {
    return (num * 100).toFixed(decimals) + '%';
}

// Show loading state for elements
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    }
}

// Hide loading state for elements
function hideLoading(elementId, content) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = content;
    }
}

// Error handling for API calls
function handleApiError(error, context = '') {
    console.error(`Erro na API ${context}:`, error);
    
    // Show user-friendly error message
    const errorMessage = `Erro ao carregar dados${context ? ' de ' + context : ''}. Tente recarregar a página.`;
    
    // You could show a toast notification here
    console.warn(errorMessage);
}

// Debounce function for search/filter inputs
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
