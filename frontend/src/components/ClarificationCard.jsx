import { useState } from 'react';
import './ClarificationCard.css';

export default function ClarificationCard({ questions, onAnswer, onSkip }) {
  const [answer, setAnswer] = useState('');

  return (
    <div className="clarification-card">
      <h4>The council has some clarifying questions:</h4>
      <ol>
        {questions.map((q, i) => (
          <li key={i}>{q}</li>
        ))}
      </ol>
      <textarea
        placeholder="Your answer (optional)…"
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
      />
      <div className="card-actions">
        <button
          className="btn-answer"
          disabled={!answer.trim()}
          onClick={() => onAnswer(answer.trim())}
        >
          Answer
        </button>
        <button className="btn-skip" onClick={onSkip}>
          Skip
        </button>
      </div>
    </div>
  );
}
