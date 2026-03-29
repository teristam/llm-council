import ReactMarkdown from 'react-markdown';
import './Stage2_5.css';

export default function Stage2_5({ devilAdvocate }) {
  if (!devilAdvocate) {
    return null;
  }

  const modelShortName = devilAdvocate.model.split('/')[1] || devilAdvocate.model;

  return (
    <div className="stage stage2_5">
      <h3 className="stage-title">Stage 2.5: Devil's Advocate</h3>
      <p className="stage-description">
        This model was instructed to find where the other models agreed — and argue against it.
      </p>

      <div className="da-content">
        <div className="da-model-label">Devil's Advocate: {modelShortName}</div>

        {devilAdvocate.consensus_identified && (
          <div className="da-section">
            <div className="da-section-title">Consensus Identified</div>
            <div className="da-section-text markdown-content">
              <ReactMarkdown>{devilAdvocate.consensus_identified}</ReactMarkdown>
            </div>
          </div>
        )}

        <div className="da-section">
          <div className="da-section-title">Critique</div>
          <div className="da-section-text markdown-content">
            <ReactMarkdown>{devilAdvocate.critique || devilAdvocate.raw}</ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
}
