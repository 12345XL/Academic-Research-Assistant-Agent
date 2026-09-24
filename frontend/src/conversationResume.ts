import type { Conversation, System } from './api';

// Same-tab reload only. No bearer token, question, answer, or cached grant.
export const RESUME_KEY = 'research-agent-conversation-resume-v1';
export interface ConversationPointer {
  conversation_id: string;
  paper_id: string;
  principal_id: string;
  mode: 'local_public' | 'bearer_policy';
  remember: boolean;
}

export function readConversationPointer(): ConversationPointer | null {
  try {
    const raw = sessionStorage.getItem(RESUME_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw);
    if (value && typeof value.conversation_id === 'string' && /^[0-9a-f]{32}$/.test(value.conversation_id)
        && typeof value.paper_id === 'string' && value.paper_id.length > 0 && value.paper_id.length <= 150
        && typeof value.principal_id === 'string' && value.principal_id.length > 0 && value.principal_id.length <= 100
        && ['local_public', 'bearer_policy'].includes(value.mode) && typeof value.remember === 'boolean') {
      return { conversation_id: value.conversation_id, paper_id: value.paper_id,
        principal_id: value.principal_id, mode: value.mode, remember: value.remember };
    }
    sessionStorage.removeItem(RESUME_KEY);
  } catch { clearConversationPointer(); }
  return null;
}

export function clearConversationPointer(): void {
  try { sessionStorage.removeItem(RESUME_KEY); } catch { /* storage may be disabled */ }
}

export function saveConversationPointer(conversation: Conversation, access: System['access'], remember: boolean): boolean {
  if (!access?.principal_id) return false;
  const pointer: ConversationPointer = { conversation_id: conversation.conversation_id,
    paper_id: conversation.paper_id, principal_id: access.principal_id, mode: access.mode, remember };
  try { sessionStorage.setItem(RESUME_KEY, JSON.stringify(pointer)); return true; }
  catch { return false; }
}
