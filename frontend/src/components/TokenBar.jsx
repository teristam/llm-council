import './TokenBar.css';

export default function TokenBar({ totalTokens }) {
  const display = totalTokens == null
    ? '—'
    : totalTokens.toLocaleString();

  return (
    <div className="token-bar">
      Tokens used this request: <span className="token-count">{display}</span>
    </div>
  );
}
