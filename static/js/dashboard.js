// Dashboard JavaScript for Interactive Charts

let charts = {};
let originalData = {};

document.addEventListener('DOMContentLoaded', function() {
    // Load all dashboard data
    loadDashboardData();
    
    // Initialize year filters
    initializeYearFilters();
});

// Load all dashboard data
async function loadDashboardData() {
    try {
        // Load summary data
        await loadSummaryCards();
        
        // Load chart data
        await Promise.all([
            loadCasosPorAno(),
            loadDistribuicaoSexo(),
            loadCasosPorMes(),
            loadFenomenoClimatico(),
            loadHospitalizacaoIdade(),
            loadRacaDistribution() // Adicionado
        ]);
        
    } catch (error) {
        console.error('Erro ao carregar dados do dashboard:', error);
    }
}

// Load summary cards data
async function loadSummaryCards() {
    try {
        const response = await fetch('/api/data/summary');
        const data = await response.json();
        
        document.getElementById('total-casos-dash').textContent = data.total_casos.toLocaleString('pt-BR');
        document.getElementById('casos-hosp-dash').textContent = data.casos_hospitalizados.toLocaleString('pt-BR');
        document.getElementById('taxa-hosp-dash').textContent = data.taxa_hospitalizacao.toFixed(2) + '%';
        document.getElementById('idade-media-dash').textContent = Math.round(data.idade_media) + ' anos';
        
    } catch (error) {
        console.error('Erro ao carregar resumo:', error);
    }
}

// Load casos por ano chart
async function loadCasosPorAno() {
    try {
        const response = await fetch('/api/data/casos_por_ano');
        const data = await response.json();
        originalData.casosPorAno = data;
        
        const ctx = document.getElementById('casosPorAnoChart').getContext('2d');
        charts.casosPorAno = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.anos,
                datasets: [{
                    label: 'Casos de Dengue',
                    data: data.casos,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Erro ao carregar casos por ano:', error);
    }
}

// Update casos por ano chart type
function updateCasosPorAno(type) {
    if (charts.casosPorAno) {
        charts.casosPorAno.config.type = type;
        charts.casosPorAno.update();
        
        // Update button states
        document.querySelectorAll('.btn-group button').forEach(btn => {
            btn.classList.remove('active');
        });
        event.target.classList.add('active');
    }
}

// Load distribuição por sexo chart
async function loadDistribuicaoSexo() {
    try {
        const response = await fetch('/api/data/distribuicao_sexo');
        const data = await response.json();
        originalData.distribuicaoSexo = data;
        
        const ctx = document.getElementById('distribuicaoSexoChart').getContext('2d');
        charts.distribuicaoSexo = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: [
                        '#f093fb',
                        '#667eea'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Erro ao carregar distribuição por sexo:', error);
    }
}

// Load casos por mês chart
async function loadCasosPorMes() {
    try {
        const response = await fetch('/api/data/casos_por_mes');
        const data = await response.json();
        originalData.casosPorMes = data;
        
        const ctx = document.getElementById('casosPorMesChart').getContext('2d');
        charts.casosPorMes = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.meses,
                datasets: [{
                    label: 'Casos por Mês',
                    data: data.casos,
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: '#667eea',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Erro ao carregar casos por mês:', error);
    }
}

// Load fenômeno climático chart
async function loadFenomenoClimatico() {
    try {
        const response = await fetch('/api/data/fenomeno_climatico');
        const data = await response.json();
        originalData.fenomenoClimatico = data;
        
        const ctx = document.getElementById('fenomenoClimaticoChart').getContext('2d');
        charts.fenomenoClimatico = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: [
                        '#ffecd2',
                        '#a8edea',
                        '#f093fb'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Erro ao carregar fenômeno climático:', error);
    }
}

// Load hospitalização por idade chart
async function loadHospitalizacaoIdade() {
    try {
        const response = await fetch('/api/data/hospitalizacao_por_idade');
        const data = await response.json();
        originalData.hospitalizacaoIdade = data;
        
        const ctx = document.getElementById('hospitalizacaoIdadeChart').getContext('2d');
        charts.hospitalizacaoIdade = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.faixas,
                datasets: [{
                    label: 'Hospitalizados',
                    data: data.hospitalizados,
                    backgroundColor: 'rgba(220, 53, 69, 0.8)',
                    borderColor: '#dc3545',
                    borderWidth: 1
                }, {
                    label: 'Total de Casos',
                    data: data.total,
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: '#667eea',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Erro ao carregar hospitalização por idade:', error);
    }
}

// Load distribuição por raça chart
async function loadRacaDistribution() {
    try {
        const response = await fetch('/api/data/distribuicao_raca');
        const data = await response.json();
        originalData.racaDistribution = data;
        
        const ctx = document.getElementById('racaDistributionChart').getContext('2d');
        charts.racaDistribution = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Distribuição por Raça',
                    data: data.values,
                    backgroundColor: 'rgba(255, 159, 64, 0.8)',
                    borderColor: '#ff9f40',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Erro ao carregar distribuição por raça:', error);
    }
}

// Update hospitalização por idade chart
function updateHospitalizacaoIdade() {
    const showPercentage = document.getElementById('showPercentage').checked;
    const data = originalData.hospitalizacaoIdade;
    
    if (showPercentage && data) {
        // Calculate percentages
        const percentages = data.hospitalizados.map((hosp, index) => {
            return data.total[index] > 0 ? (hosp / data.total[index] * 100).toFixed(2) : 0;
        });
        
        charts.hospitalizacaoIdade.data.datasets = [{
            label: 'Taxa de Hospitalização (%)',
            data: percentages,
            backgroundColor: 'rgba(255, 193, 7, 0.8)',
            borderColor: '#ffc107',
            borderWidth: 1
        }];
        
        charts.hospitalizacaoIdade.options.scales.y.max = 100;
    } else {
        // Show absolute numbers
        charts.hospitalizacaoIdade.data.datasets = [{
            label: 'Hospitalizados',
            data: data.hospitalizados,
            backgroundColor: 'rgba(220, 53, 69, 0.8)',
            borderColor: '#dc3545',
            borderWidth: 1
        }, {
            label: 'Total de Casos',
            data: data.total,
            backgroundColor: 'rgba(102, 126, 234, 0.8)',
            borderColor: '#667eea',
            borderWidth: 1
        }];
        
        delete charts.hospitalizacaoIdade.options.scales.y.max;
    }
    
    charts.hospitalizacaoIdade.update();
}

// Initialize year filters
function initializeYearFilters() {
    const anoInicial = document.getElementById('anoInicial');
    const anoFinal = document.getElementById('anoFinal');
    
    // Populate year options (2000-2025)
    for (let year = 2000; year <= 2025; year++) {
        const option1 = new Option(year, year);
        const option2 = new Option(year, year);
        anoInicial.appendChild(option1);
        anoFinal.appendChild(option2);
    }
}

// Apply filters
async function applyFilters() {
    const filters = {
        anoInicial: document.getElementById('anoInicial').value,
        anoFinal: document.getElementById('anoFinal').value,
        fenomeno: document.getElementById('fenomenoFiltro').value,
        sexo: document.getElementById('sexoFiltro').value
    };
    
    try {
        const response = await fetch('/api/data/filtered', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(filters)
        });
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        console.error('Erro ao aplicar filtros:', error);
    }
}

// Clear filters
function clearFilters() {
    document.getElementById('anoInicial').value = '';
    document.getElementById('anoFinal').value = '';
    document.getElementById('fenomenoFiltro').value = '';
    document.getElementById('sexoFiltro').value = '';
    
    // Reload original data
    loadDashboardData();
}

// Update all dashboard components with new data
function updateDashboard(data) {
    // Update summary cards
    document.getElementById('total-casos-dash').textContent = data.summary.total_casos.toLocaleString('pt-BR');
    document.getElementById('casos-hosp-dash').textContent = data.summary.casos_hospitalizados.toLocaleString('pt-BR');
    document.getElementById('taxa-hosp-dash').textContent = data.summary.taxa_hospitalizacao.toFixed(2) + '%';
    document.getElementById('idade-media-dash').textContent = Math.round(data.summary.idade_media) + ' anos';

    // Update charts
    updateChart(charts.casosPorAno, data.casosPorAno.anos, [data.casosPorAno.casos]);
    updateChart(charts.distribuicaoSexo, data.distribuicaoSexo.labels, [data.distribuicaoSexo.values]);
    updateChart(charts.casosPorMes, data.casosPorMes.meses, [data.casosPorMes.casos]);
    updateChart(charts.fenomenoClimatico, data.fenomenoClimatico.labels, [data.fenomenoClimatico.values]);
    updateChart(charts.hospitalizacaoIdade, data.hospitalizacaoIdade.faixas, [data.hospitalizacaoIdade.hospitalizados, data.hospitalizacaoIdade.total]);
    updateChart(charts.racaDistribution, data.racaDistribution.labels, [data.racaDistribution.values]);
}

// Utility function to update a chart
function updateChart(chart, labels, datasetsData) {
    if (chart) {
        chart.data.labels = labels;
        chart.data.datasets.forEach((dataset, i) => {
            dataset.data = datasetsData[i];
        });
        chart.update();
    }
}
