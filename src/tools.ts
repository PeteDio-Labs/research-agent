/**
 * Deterministic step plan for research-agent.
 *
 * 1. web-search   — fetch results from MC web-search proxy
 * 2. rag-query    — fetch RAG chunks from blog index (skipped if !includeBlogKnowledge)
 * 3. synthesize   — Ollama (petedio-writer) fuses web + RAG into an answer
 *
 * Step state (search results, RAG chunks) is shared across steps via a
 * mutable ResearchState passed to the executeStep factory.
 */

import type { ResearchAgentInput } from './schema.js';
import {
  searchWeb,
  queryRag,
  ollamaChat,
  type SearchResult,
  type RagChunk,
} from './clients.js';

export type ResearchAction = 'web-search' | 'rag-query' | 'synthesize';

export interface ResearchStep {
  title: string;
  action: ResearchAction;
  args?: Record<string, unknown>;
}

export interface ResearchStepLog {
  step: ResearchStep;
  status: 'complete' | 'failed' | 'skipped';
  output: string;
  startedAt: string;
  completedAt: string;
  durationMs: number;
}

export interface ResearchState {
  webResults: SearchResult[];
  ragChunks: RagChunk[];
  answer: string;
}

export function createInitialState(): ResearchState {
  return { webResults: [], ragChunks: [], answer: '' };
}

// ─── Plan builder ─────────────────────────────────────────────────

export function buildPlan(input: ResearchAgentInput): ResearchStep[] {
  const maxResults = input.depth === 'deep' ? 10 : 5;
  const steps: ResearchStep[] = [
    {
      title: `Web search (${input.depth}, max ${maxResults})`,
      action: 'web-search',
      args: { query: input.query, maxResults },
    },
    {
      title: `RAG query (topK=${input.ragTopK})`,
      action: 'rag-query',
      args: { query: input.query, topK: input.ragTopK },
    },
    {
      title: 'Synthesize answer (petedio-writer)',
      action: 'synthesize',
      args: { query: input.query },
    },
  ];
  return steps;
}

// ─── Step executor (factory closes over per-run state) ────────────

export function createExecuteStep(
  input: ResearchAgentInput,
  state: ResearchState,
): (step: ResearchStep) => Promise<string> {
  return async (step) => {
    switch (step.action) {
      case 'web-search': {
        const query = String(step.args?.query ?? input.query);
        const maxResults = Number(step.args?.maxResults ?? 5);
        const response = await searchWeb(query, maxResults);
        state.webResults = response.results;
        if (response.results.length === 0) {
          return `No results from web search (provider=${response.provider}).`;
        }
        return [
          `Provider: ${response.provider} (${response.results.length} results, ${response.metadata.durationMs}ms)`,
          ...response.results.map(
            (r, i) => `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.snippet}`,
          ),
        ].join('\n');
      }

      case 'rag-query': {
        const query = String(step.args?.query ?? input.query);
        const topK = Number(step.args?.topK ?? input.ragTopK);
        const response = await queryRag(query, topK);
        state.ragChunks = response.results;
        if (response.count === 0) {
          return 'No RAG chunks matched.';
        }
        return [
          `Retrieved ${response.count} chunk(s):`,
          ...response.results.map(
            (c, i) =>
              `${i + 1}. [${c.sourceType}/${c.sourceRef}] sim=${c.similarity.toFixed(3)}\n   ${truncate(c.chunkText, 300)}`,
          ),
        ].join('\n');
      }

      case 'synthesize': {
        const query = String(step.args?.query ?? input.query);
        const webBlock = state.webResults.length
          ? state.webResults
              .map((r, i) => `[W${i + 1}] ${r.title} (${r.url})\n${r.snippet}`)
              .join('\n\n')
          : '(no web results)';
        const ragBlock = state.ragChunks.length
          ? state.ragChunks
              .map(
                (c, i) =>
                  `[R${i + 1}] ${c.sourceType}/${c.sourceRef}\n${truncate(c.chunkText, 600)}`,
              )
              .join('\n\n')
          : '(no blog knowledge)';

        const system = [
          'You are PeteDio research-agent. Synthesize the question using the supplied sources.',
          'Cite sources inline as [W1], [W2] for web and [R1], [R2] for blog/RAG.',
          'Prefer concrete facts. If sources disagree, say so. If sources are insufficient, say so.',
          'Output: 3-7 sentence answer, then a "Sources" section listing only the citations actually used.',
        ].join(' ');

        const user = [
          `Question: ${query}`,
          '',
          'Web results:',
          webBlock,
          '',
          'Blog knowledge:',
          ragBlock,
        ].join('\n');

        const answer = await ollamaChat([
          { role: 'system', content: system },
          { role: 'user', content: user },
        ]);
        state.answer = answer;
        return answer;
      }

      default:
        throw new Error(`Unknown research action: ${(step as ResearchStep).action}`);
    }
  };
}

// ─── Report formatter ─────────────────────────────────────────────

export function formatReport(
  input: ResearchAgentInput,
  state: ResearchState,
  logs: ResearchStepLog[],
): string {
  const lines: string[] = [
    `# Research: ${input.query}`,
    '',
    `Depth: ${input.depth} · RAG topK: ${input.ragTopK} · Blog knowledge: ${input.includeBlogKnowledge ? 'on' : 'off'}`,
    '',
  ];
  if (state.answer) {
    lines.push('## Answer', '', state.answer, '');
  }
  lines.push('## Steps', '');
  for (const [i, log] of logs.entries()) {
    lines.push(`${i + 1}. **${log.step.title}** [${log.status}, ${log.durationMs}ms]`);
    if (log.output && log.step.action !== 'synthesize') {
      lines.push('', '```', log.output, '```', '');
    }
  }
  return lines.join('\n');
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max) + '…';
}
