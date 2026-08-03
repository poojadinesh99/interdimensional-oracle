const questionInput = document.getElementById('question');
const askButton = document.getElementById('ask');
const output = document.getElementById('output');

askButton.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  if (!question) {
    output.textContent = 'Please enter a question.';
    return;
  }

  output.textContent = 'Consulting the oracle…';

  try {
    const response = await fetch('/api/oracle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    output.textContent = data.ok ? data.answer : data.error;
  } catch (error) {
    output.textContent = `Request failed: ${error.message}`;
  }
});
