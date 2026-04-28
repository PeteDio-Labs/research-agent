import { z } from 'zod';
import { TaskPayloadSchema } from '@petedio/shared/agents';

export const ResearchAgentInputSchema = z.object({
  query: z.string().min(1).describe('The research question'),
  depth: z.enum(['quick', 'deep']).default('quick')
    .describe('quick=5 web results, deep=10 web results'),
  ragTopK: z.number().int().min(1).max(20).default(5)
    .describe('Number of RAG chunks to retrieve from blog index'),
  includeBlogKnowledge: z.boolean().default(true)
    .describe('When false, skips the RAG step and synthesizes from web only'),
});

export type ResearchAgentInput = z.infer<typeof ResearchAgentInputSchema>;

export const ResearchTaskPayloadSchema = TaskPayloadSchema.extend({
  input: ResearchAgentInputSchema,
});
