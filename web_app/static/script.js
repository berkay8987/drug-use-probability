// Mappings (Must match predict.py)
const MAPPINGS = {
    age: {
        '18-24': -0.95197, '25-34': -0.07854, '35-44': 0.49788,
        '45-54': 1.09449, '55-64': 1.82213, '65+': 2.59171
    },
    gender: {
        'Female': 0.48246, 'Male': -0.48246
    },
    education: {
        'Left school before 16 years': -2.43591,
        'Left school at 16 years': -1.73790,
        'Left school at 17 years': -1.43719,
        'Left school at 18 years': -1.22751,
        'Some college or university, no certificate or degree': -0.61113,
        'Professional certificate/ diploma': -0.05921,
        'University degree': 0.45468,
        'Masters degree': 1.16365,
        'Doctorate degree': 1.98437
    },
    country: {
        'Australia': -0.09765, 'Canada': 0.24923, 'New Zealand': -0.46841,
        'Other': -0.28519, 'Republic of Ireland': 0.21128, 'UK': 0.96082, 'USA': -0.57009
    },
    ethnicity: {
        'Asian': -0.50212, 'Black': -1.10702, 'Mixed-Black/Asian': 1.90725,
        'Mixed-White/Asian': 0.12600, 'Mixed-White/Black': -0.22166,
        'Other': 0.11440, 'White': -0.31685
    },
    nscore: { 'Very Low': -2.5, 'Low': -1.0, 'Average': 0.0, 'High': 1.0, 'Very High': 2.5 },
    escore: { 'Very Low': -2.5, 'Low': -1.0, 'Average': 0.0, 'High': 1.0, 'Very High': 2.5 },
    oscore: { 'Very Low': -2.5, 'Low': -1.0, 'Average': 0.0, 'High': 1.0, 'Very High': 2.5 },
    ascore: { 'Very Low': -2.5, 'Low': -1.0, 'Average': 0.0, 'High': 1.0, 'Very High': 2.5 },
    cscore: { 'Very Low': -2.5, 'Low': -1.0, 'Average': 0.0, 'High': 1.0, 'Very High': 2.5 },
    impulsive: { 'Very Low': -2.55, 'Low': -1.37, 'Average': -0.21, 'High': 0.52, 'Very High': 1.86 },
    ss: { 'Very Low': -2.07, 'Low': -1.18, 'Average': -0.21, 'High': 0.40, 'Very High': 1.22 }
};

const QUESTIONS = [
    { id: 'age', title: 'What is your age group?', options: Object.keys(MAPPINGS.age) },
    { id: 'gender', title: 'What is your gender?', options: Object.keys(MAPPINGS.gender) },
    { id: 'education', title: 'Highest education level?', options: Object.keys(MAPPINGS.education) },
    { id: 'country', title: 'Country of residence?', options: Object.keys(MAPPINGS.country) },
    { id: 'ethnicity', title: 'Ethnicity?', options: Object.keys(MAPPINGS.ethnicity) },
    { id: 'nscore', title: 'Neuroticism (prone to worry/anxiety)?', options: Object.keys(MAPPINGS.nscore) },
    { id: 'escore', title: 'Extraversion (sociable/active)?', options: Object.keys(MAPPINGS.escore) },
    { id: 'oscore', title: 'Openness to Experience (imaginative)?', options: Object.keys(MAPPINGS.oscore) },
    { id: 'ascore', title: 'Agreeableness (trusting/cooperative)?', options: Object.keys(MAPPINGS.ascore) },
    { id: 'cscore', title: 'Conscientiousness (organized)?', options: Object.keys(MAPPINGS.cscore) },
    { id: 'impulsive', title: 'Impulsivity (acting without thinking)?', options: Object.keys(MAPPINGS.impulsive) },
    { id: 'ss', title: 'Sensation Seeking (thrill seeking)?', options: Object.keys(MAPPINGS.ss) }
];

let currentQuestionIndex = 0;
let answers = {};

document.addEventListener('DOMContentLoaded', () => {
    renderQuestion();

    document.getElementById('prev-btn').addEventListener('click', () => {
        if (currentQuestionIndex > 0) {
            currentQuestionIndex--;
            renderQuestion();
        }
    });

    document.getElementById('restart-btn').addEventListener('click', () => {
        currentQuestionIndex = 0;
        answers = {};
        document.getElementById('results-container').style.display = 'none';
        document.getElementById('wizard-container').style.display = 'block';
        renderQuestion();
    });
});

function renderQuestion() {
    const q = QUESTIONS[currentQuestionIndex];
    const questionArea = document.getElementById('question-area');
    const title = document.getElementById('question-title');
    const optionsContainer = document.getElementById('options-container');
    const prevBtn = document.getElementById('prev-btn');
    const progressBar = document.getElementById('progress-bar');


    const progress = ((currentQuestionIndex) / QUESTIONS.length) * 100;
    progressBar.style.width = `${progress}%`;

    questionArea.classList.add('fade-out');

    setTimeout(() => {
        title.innerText = q.title;
        optionsContainer.innerHTML = '';

        q.options.forEach(opt => {
            const btn = document.createElement('button');
            btn.className = 'option-btn';
            btn.innerText = opt;
            btn.onclick = () => handleAnswer(q.id, opt);
            optionsContainer.appendChild(btn);
        });

        prevBtn.style.display = currentQuestionIndex === 0 ? 'none' : 'inline-block';

        questionArea.classList.remove('fade-out');
        questionArea.classList.add('fade-in');
        setTimeout(() => questionArea.classList.remove('fade-in'), 300);
    }, 300);
}

function handleAnswer(key, value) {
    answers[key] = MAPPINGS[key][value];

    if (currentQuestionIndex < QUESTIONS.length - 1) {
        currentQuestionIndex++;
        renderQuestion();
    } else {
        submitAnswers();
    }
}

async function submitAnswers() {
    document.getElementById('wizard-container').style.display = 'none';
    document.getElementById('loading').style.display = 'block';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(answers)
        });

        const results = await response.json();

        if (response.ok) {
            renderResults(results);
        } else {
            alert('Error: ' + results.error);
            document.getElementById('loading').style.display = 'none';
            document.getElementById('wizard-container').style.display = 'block';
        }
    } catch (error) {
        alert('Network Error: ' + error);
        document.getElementById('loading').style.display = 'none';
        document.getElementById('wizard-container').style.display = 'block';
    }
}

function renderResults(results) {
    document.getElementById('loading').style.display = 'none';
    const container = document.getElementById('results-container');
    const tbody = document.querySelector('#results-table tbody');

    tbody.innerHTML = '';

    results.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${r.drug}</td>
            <td>${r.risk}</td>
            <td>${r.confidence}%</td>
        `;
        tbody.appendChild(tr);
    });

    container.style.display = 'block';
}
