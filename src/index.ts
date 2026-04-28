/**
 * research-agent — Web + blog-RAG research agent.
 *
 * Accepts a TaskPayload from MC Backend, runs a deterministic 3-step plan
 * (web-search → rag-query → synthesize), and reports the synthesized answer
 * back to MC. Skips rag-query when input.includeBlogKnowledge is false.
 */

import express from 'express';
import pino from 'pino';
import { z } from 'zod';
import { ResearchAgentInputSchema } from './schema.js';
import {
  buildPlan,
  createExecuteStep,
  createInitialState,
  formatReport,
  type ResearchStep,
  type ResearchStepLog,
} from './tools.js';

const log = pino({ level: process.env.LOG_LEVEL ?? 'info' });
const PORT = parseInt(process.env.PORT ?? '3010', 10);
const MC_BACKEND_URL = process.env.MC_BACKEND_URL ?? 'http://localhost:3000';
const SHARED_AGENTS_MODULE_PATH = process.env.SHARED_AGENTS_MODULE_PATH ?? '@petedio/shared/agents';

interface SharedAgentReporter {
  running(message: string): Promise<void>;
  complete(result: {
    taskId: string;
    agentName: string;
    status: 'complete';
    summary: string;
    artifacts: Array<{ type: string; label: string; content: string }>;
    durationMs: number;
    completedAt: string;
  }): Promise<void>;
  fail(message: string): Promise<void>;
}

interface SharedAgentsModule {
  AgentReporter: new (opts: { mcUrl: string; taskId: string; agentName: string }) => SharedAgentReporter;
  TaskPayloadSchema: z.ZodType<{
    taskId: string;
    agentName: string;
    trigger: string;
    input: Record<string, unknown>;
    issuedAt: string;
  }>;
  runDeterministicPlan: (opts: {
    steps: ResearchStep[];
    executeStep: (step: ResearchStep) => Promise<string>;
    onBeforeStep?: (step: ResearchStep, index: number) => Promise<'proceed' | 'skip' | 'abort'>;
    onStepStart?: (step: ResearchStep, index: number) => void | Promise<void>;
    stopOnError?: boolean;
  }) => Promise<{
    status: 'complete' | 'failed';
    logs: ResearchStepLog[];
    completedSteps: number;
    skippedSteps: number;
    failedStep?: ResearchStepLog;
  }>;
}

async function loadSharedAgents(): Promise<SharedAgentsModule> {
  return import(SHARED_AGENTS_MODULE_PATH) as Promise<SharedAgentsModule>;
}

// ─── Agent Logic ─────────────────────────────────────────────────

async function runResearch(payload: { taskId: string; input: Record<string, unknown> }): Promise<void> {
  const startMs = Date.now();
  const input = ResearchAgentInputSchema.parse(payload.input);
  const shared = await loadSharedAgents();
  const { AgentReporter, runDeterministicPlan } = shared;

  const reporter = new AgentReporter({
    mcUrl: MC_BACKEND_URL,
    taskId: payload.taskId,
    agentName: 'research-agent',
  });

  await reporter.running(`Starting research: "${input.query}" (${input.depth})...`);
  log.info({ taskId: payload.taskId, input }, 'research-agent starting');

  const steps = buildPlan(input);
  const state = createInitialState();
  const executeStep = createExecuteStep(input, state);

  try {
    const result = await runDeterministicPlan({
      steps,
      executeStep,
      onBeforeStep: async (step) => {
        if (step.action === 'rag-query' && !input.includeBlogKnowledge) {
          return 'skip';
        }
        return 'proceed';
      },
      onStepStart: async (step, index) => {
        await reporter.running(`Step ${index + 1}/${steps.length}: ${step.title}`);
      },
    });

    const durationMs = Date.now() - startMs;
    const report = formatReport(input, state, result.logs);
    const summary = result.failedStep
      ? `Failed at: ${result.failedStep.step.title}`
      : `Research complete — ${result.completedSteps} step(s) finished${result.skippedSteps ? `, ${result.skippedSteps} skipped` : ''}`;

    log.info(
      { taskId: payload.taskId, durationMs, steps: result.logs.length, status: result.status },
      'research complete',
    );

    if (result.status === 'failed') {
      await reporter.fail(`${summary}\n\n${report}`);
      return;
    }

    await reporter.complete({
      taskId: payload.taskId,
      agentName: 'research-agent',
      status: 'complete',
      summary,
      artifacts: [
        {
          type: 'research-report',
          label: `Research: ${input.query}`,
          content: report,
        },
      ],
      durationMs,
      completedAt: new Date().toISOString(),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    log.error({ taskId: payload.taskId, err: msg }, 'research failed');
    await reporter.fail(msg);
  }
}

// ─── HTTP Server ──────────────────────────────────────────────────

const app = express();
app.use(express.json({ limit: '1mb' }));

const shared = await loadSharedAgents();
const { TaskPayloadSchema } = shared;

app.post('/run', async (req, res) => {
  const parsed = TaskPayloadSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: 'Invalid task payload', details: parsed.error.flatten() });
    return;
  }

  res.json({ accepted: true, taskId: parsed.data.taskId });

  runResearch(parsed.data).catch((err) => {
    log.error({ err: err instanceof Error ? err.message : err }, 'Unhandled research error');
  });
});

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    agent: 'research-agent',
    sharedAgentsModulePath: SHARED_AGENTS_MODULE_PATH,
    ollamaModel: process.env.OLLAMA_MODEL ?? 'petedio-writer',
  });
});

app.listen(PORT, () => {
  log.info({ port: PORT, sharedAgentsModulePath: SHARED_AGENTS_MODULE_PATH }, 'research-agent listening');
});
