import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import { api } from './api';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [tokenTotal, setTokenTotal] = useState(null);
  const [pendingClarification, setPendingClarification] = useState(null);
  // Shape: { conversationId, userQuery, stage1Results, questionsByModel, consolidatedQuestions, asstMessageId }

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, []);

  // Load conversation details when selected
  useEffect(() => {
    if (currentConversationId) {
      loadConversation(currentConversationId);
    }
  }, [currentConversationId]);

  const loadConversations = async () => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadConversation = async (id) => {
    try {
      const conv = await api.getConversation(id);
      setCurrentConversation(conv);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleNewConversation = async () => {
    try {
      const newConv = await api.createConversation();
      setConversations([
        { id: newConv.id, created_at: newConv.created_at, message_count: 0 },
        ...conversations,
      ]);
      setCurrentConversationId(newConv.id);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    setCurrentConversationId(id);
  };

  const triggerPass2 = async ({ conversationId, userQuery, stage1Results, questionsByModel, consolidatedQuestions, userAnswer, asstMessageId = null }) => {
    setPendingClarification(null);
    setIsLoading(true);

    const payload = {
      user_answer: userAnswer,
      user_query: userQuery,
      stage1_results: stage1Results,
      questions_by_model: questionsByModel,
      consolidated_questions: consolidatedQuestions,
      asst_message_id: asstMessageId,
    };

    try {
      await api.clarifyMessageStream(conversationId, payload, (eventType, event) => {
        switch (eventType) {
          case 'stage1c_start':
          case 'stage2_start':
          case 'stage2_5_start':
          case 'stage3_start': {
            const stageKey = eventType.replace('_start', '').replace('stage1c', 'stage1');
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.loading = { ...lastMsg.loading, [stageKey]: true };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            break;
          }

          case 'stage1c_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.stage1 = event.data;
              lastMsg.loading = { ...lastMsg.loading, stage1: false };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
            break;

          case 'stage2_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.stage2 = event.data;
              lastMsg.metadata = event.metadata;
              lastMsg.loading = { ...lastMsg.loading, stage2: false };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
            break;

          case 'stage2_5_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.stage2_5 = event.data;
              lastMsg.loading = { ...lastMsg.loading, stage2_5: false };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
            break;

          case 'stage3_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.stage3 = event.data;
              lastMsg.loading = { ...lastMsg.loading, stage3: false };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
            break;

          case 'complete':
            loadConversations();
            setIsLoading(false);
            break;

          case 'error':
            console.error('Clarify stream error:', event.message);
            setIsLoading(false);
            break;

          case 'stream_end':
            setIsLoading(false);
            break;

          case 'keepalive':
            break;

          default:
            console.log('Unknown clarify event:', eventType);
        }
      });
    } catch (error) {
      console.error('Failed to clarify:', error);
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (content) => {
    if (!currentConversationId) return;

    setIsLoading(true);
    setTokenTotal(null);
    try {
      // Optimistically add user message to UI
      const userMessage = { role: 'user', content };
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      // Create a partial assistant message that will be updated progressively
      const assistantMessage = {
        role: 'assistant',
        stage1: null,
        stage2: null,
        stage2_5: null,
        stage3: null,
        metadata: null,
        loading: {
          stage1: false,
          stage2: false,
          stage2_5: false,
          stage3: false,
        },
      };

      // Add the partial assistant message
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
      }));

      // Send message with streaming
      await api.sendMessageStream(currentConversationId, content, (eventType, event) => {
        switch (eventType) {
          case 'stage1_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage1 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage1_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage1 = event.data;
              lastMsg.loading.stage1 = false;
              return { ...prev, messages };
            });
            setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
            break;

          case 'stage1b_start':
            // no UI update needed
            break;

          case 'clarification_needed':
            setPendingClarification({
              conversationId: currentConversationId,
              userQuery: content,
              stage1Results: event.stage1_results,
              questionsByModel: event.questions_by_model,
              consolidatedQuestions: event.questions,
              asstMessageId: null,
            });
            setIsLoading(false);
            break;

          case 'clarification_skipped':
            triggerPass2({
              conversationId: currentConversationId,
              userQuery: content,
              stage1Results: event.stage1_results,
              questionsByModel: event.questions_by_model,
              consolidatedQuestions: [],
              userAnswer: null,
              asstMessageId: null,
            });
            break;

          case 'title_complete':
            // Reload conversations to get updated title
            loadConversations();
            break;

          case 'error':
            console.error('Stream error:', event.message);
            setIsLoading(false);
            break;

          case 'stream_end':
            // Safety net: stream ended, ensure loading is cleared
            setIsLoading(false);
            break;

          case 'keepalive':
            // Keepalive heartbeat from server - ignore silently
            break;

          default:
            console.log('Unknown event type:', eventType);
        }
      });
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove optimistic messages on error
      setCurrentConversation((prev) => ({
        ...prev,
        messages: prev.messages.slice(0, -2),
      }));
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
      />
      <ChatInterface
        conversation={currentConversation}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
        tokenTotal={tokenTotal}
        pendingClarification={pendingClarification}
        onClarificationAnswer={(answer) => triggerPass2({ ...pendingClarification, userAnswer: answer })}
        onClarificationSkip={() => triggerPass2({ ...pendingClarification, userAnswer: null })}
      />
    </div>
  );
}

export default App;
