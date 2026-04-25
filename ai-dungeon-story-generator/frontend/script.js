// State Management
let currentContext = null;
let storyHistory = "";

// DOM Elements
const setupForm = document.getElementById('setup-form');
const startBtn = document.getElementById('start-btn');
const startSpinner = document.getElementById('start-spinner');
const setupScreen = document.getElementById('setup-screen');
const storyScreen = document.getElementById('story-screen');
const storyHistoryContainer = document.getElementById('story-history');
const choicesContainer = document.getElementById('choices-container');
const loadingIndicator = document.getElementById('loading-indicator');

// Event Listeners
setupForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Gather context
    currentContext = {
        genre: document.getElementById('genre').value,
        character_name: document.getElementById('char-name').value,
        character_details: document.getElementById('char-details').value,
        lore: document.getElementById('lore').value
    };

    const starterPrompt = document.getElementById('starter-prompt').value;

    // UI Feedback
    startBtn.disabled = true;
    startSpinner.style.display = 'block';

    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                context: currentContext,
                starter_prompt: starterPrompt
            })
        });

        if (!response.ok) throw new Error('Failed to generate story');

        const data = await response.json();

        // Transition UI
        setupScreen.classList.add('fade-out');
        setTimeout(() => {
            setupScreen.classList.add('hidden');
            setupScreen.classList.remove('active', 'fade-out');

            storyScreen.classList.remove('hidden');
            storyScreen.classList.add('slide-in');

            appendStoryNode(data.story_chunk, starterPrompt);
            renderChoices(data.choices);
        }, 400);

    } catch (error) {
        console.error(error);
        alert('An error occurred while generating the story.');
    } finally {
        startBtn.disabled = false;
        startSpinner.style.display = 'none';
    }
});

async function handleChoice(choiceText) {
    // UI Feedback
    choicesContainer.innerHTML = '';
    loadingIndicator.classList.remove('hidden');

    try {
        const response = await fetch('/api/continue', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                context: currentContext,
                history: storyHistory,
                user_choice: choiceText
            })
        });

        if (!response.ok) throw new Error('Failed to continue story');

        const data = await response.json();

        appendStoryNode(data.story_chunk, choiceText);
        renderChoices(data.choices);

    } catch (error) {
        console.error(error);
        alert('An error occurred while continuing the story.');
        // Recover choices on failure (ideal world would cache previous choices)
    } finally {
        loadingIndicator.classList.add('hidden');
    }
}

function appendStoryNode(storyText, previousChoice) {
    // Add to raw history
    storyHistory += `\n[Choice] ${previousChoice}\n[Story] ${storyText}`;

    // Format paragraphs
    const paragraphs = storyText.split('\n').filter(p => p.trim() !== '');
    const pHTML = paragraphs.map(p => `<p>${p}</p>`).join('');

    const nodeHTML = `
        <div class="story-node">
            ${storyHistory !== `\n[Choice] ${previousChoice}\n[Story] ${storyText}` ?
            `<span class="story-choice-label">You chose: ${previousChoice}</span>` : ''}
            <div class="story-text">
                ${pHTML}
            </div>
        </div>
    `;

    storyHistoryContainer.insertAdjacentHTML('beforeend', nodeHTML);

    // Auto-scroll to bottom of history
    storyHistoryContainer.lastElementChild.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderChoices(choices) {
    choicesContainer.innerHTML = '';

    choices.forEach((choice, index) => {
        const btn = document.createElement('button');
        btn.className = 'choice-btn';
        btn.textContent = choice;
        // Staggered animation
        btn.style.animation = `fadeIn 0.5s ease forwards ${index * 0.15}s`;
        btn.style.opacity = '0';

        btn.addEventListener('click', () => handleChoice(choice));
        choicesContainer.appendChild(btn);
    });
}
